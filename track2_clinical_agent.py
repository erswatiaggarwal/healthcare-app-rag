"""
track2_clinical_agent.py

ReAct AI Agent for the Healthcare-App Clinical Knowledge Assistant.
Five tools: semantic KB search, BM25 KB search, hybrid KB search,
mock Epic patient record lookup, and mock system/incident status check.
LLM uses build_llm() — automatically switches to Nebius AI Studio when NEBIUS_API_KEY is set.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from track2_retrieval_engine import RetrievalEngine, RetrievalResult

# ── Mock data stores ──────────────────────────────────────────────────────────

_MOCK_RECORDS: dict[str, dict] = {
    "MRN-334521": {
        "name": "J. Doe (anonymised)",
        "status": "admitted",
        "ward": "Cardiology 3B",
        "active_orders": ["metoprolol 25mg 08:00 (x2 — DUPLICATE FLAG)"],
        "flags": ["MAR-DISCREPANCY"],
    },
    "MRN-293847": {
        "name": "A. Smith (anonymised)",
        "status": "admitted",
        "ward": "General Medicine 2A",
        "active_orders": ["CBC with differential — LAB-293847-0209 (pending)"],
        "flags": [],
    },
    "MRN-847392": {
        "name": "B. Patel (anonymised)",
        "status": "awaiting admission",
        "ward": "ED",
        "active_orders": [],
        "flags": ["AUTH-4012 — pre-auth required"],
    },
}

_MOCK_STATUS: dict[str, dict] = {
    "BUG-EHR-2301": {
        "status": "active",
        "eta": "next maintenance window",
        "workaround": "pharmacy MAR review within 30 min; apply MAR-DISCREPANCY flag",
    },
    "BUG-EHR-2201": {
        "status": "active",
        "eta": "2 weeks",
        "workaround": "manual LTC entry + LTC-MANUAL-SYNC flag; notify pharmacist",
    },
    "BUG-SCH-0445": {
        "status": "active",
        "eta": "2 weeks",
        "workaround": "book via phone scheduling line (ext. 4400)",
    },
    "ERR_HL7_ROUTE_FAIL": {
        "status": "active",
        "eta": "monitoring",
        "workaround": "escalate to hl7-support@healthcare-app.internal",
    },
    "AUTH-4012": {
        "status": "operational",
        "eta": "N/A",
        "workaround": "re-submit with refreshed auth token (see ICD-ADM-001)",
    },
    "SCH-CONFLICT-88": {
        "status": "active",
        "eta": "2 weeks",
        "workaround": "book via phone scheduling line (ext. 4400)",
    },
    "TC-CONN-FAIL": {
        "status": "operational",
        "eta": "N/A",
        "workaround": "guide patient through HealthConnect device-check",
    },
}

# ── Tool factory (binds to engine instance via closure) ───────────────────────

def _make_kb_tools(engine: "RetrievalEngine") -> list[Any]:
    """Return the three KB search @tool functions bound to the given engine."""
    try:
        from langchain_core.tools import tool
    except ImportError as exc:
        raise ImportError("pip install langchain-core") from exc

    def _fmt(results: list["RetrievalResult"]) -> str:
        """Format a list of RetrievalResults as a numbered string for the agent's Observation field."""
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            preview = r.content[:200].replace("\n", " ")
            score_tag = f"score={r.score:.4f}" if r.score is not None else ""
            lines.append(
                f"{i}. [{r.doc_type}/{r.clinical_area}] {r.title}  {score_tag}\n"
                f"   {preview}..."
            )
        return "\n\n".join(lines)

    @tool
    def search_kb_semantic(query: str) -> str:
        """
        Search the Healthcare-App clinical knowledge base using semantic
        (vector) similarity. Use for conceptual or policy questions where
        the coordinator describes a situation in their own words.
        Returns top-5 results: title, doc_type, clinical_area, 200-char preview.
        """
        return _fmt(engine.search_semantic(query, k=5))

    @tool
    def search_kb_bm25(query: str) -> str:
        """
        Search the Healthcare-App clinical knowledge base using BM25 keyword
        matching. Use when the query contains exact identifiers: MRN numbers
        (MRN-XXXXXX), lab order IDs (LAB-*), encounter IDs (ENC-*), error
        codes (ERR_*), bug IDs (BUG-EHR-*, BUG-SCH-*), CPT codes, runbook
        IDs (ICD-*). Returns top-5 results as a formatted string.
        """
        return _fmt(engine.search_bm25(query, k=5))

    @tool
    def search_kb_hybrid(query: str) -> str:
        """
        Search the Healthcare-App clinical knowledge base using hybrid
        retrieval (BM25 + semantic fused via RRF). Use for mixed queries
        combining a clinical concept with a specific identifier. This is
        the most powerful mode and the default when unsure which to use.
        Returns top-5 fused results as a formatted string.
        """
        return _fmt(engine.search_hybrid(query, k=5))

    return [search_kb_semantic, search_kb_bm25, search_kb_hybrid]


def _make_system_tools() -> list[Any]:
    """Return the mock Epic and incident-status @tool functions."""
    try:
        from langchain_core.tools import tool
    except ImportError as exc:
        raise ImportError("pip install langchain-core") from exc

    @tool
    def lookup_patient_record(mrn: str) -> str:
        """
        Look up a patient's current clinical status in the Epic EHR system.
        Input: MRN in format MRN-XXXXXX.
        Returns: anonymised name, admission status, ward, active medication
        orders, and any open flags (MAR-DISCREPANCY, LTC-MANUAL-SYNC, etc.).
        NOTE: MOCK — returns synthetic data keyed on known KB MRNs.
        In production this calls the Epic FHIR API.
        """
        record = _MOCK_RECORDS.get(
            mrn.strip(),
            {"name": "Unknown patient", "status": "not found", "active_orders": [], "flags": [], "ward": ""},
        )
        return json.dumps({"mrn": mrn, **record}, indent=2)

    @tool
    def check_system_status(identifier: str) -> str:
        """
        Check the current status of a known EHR/HL7 bug or system incident.
        Input: bug ID (BUG-EHR-XXXX, BUG-SCH-XXXX) or error code
        (ERR_*, AUTH-*, SCH-CONFLICT-*, TC-CONN-FAIL).
        Returns: status (active/resolved/operational), ETA, and workaround.
        NOTE: MOCK — returns data from the static KB.
        In production this calls the live incident management API.
        """
        result = _MOCK_STATUS.get(
            identifier.strip(),
            {"status": "unknown", "eta": "N/A", "workaround": "check KB manually"},
        )
        return json.dumps({"identifier": identifier, **result}, indent=2)

    return [lookup_patient_record, check_system_status]


# ── Agent class ───────────────────────────────────────────────────────────────

class ClinicalAgent:
    """
    LangChain ReAct AgentExecutor wrapping five tools.
    Uses build_llm() so it runs against NIM in production and
    OpenAI cloud in development — no code change required.

    Usage:
        agent = ClinicalAgent(engine)
        result = agent.run("MRN-334521 has duplicate metoprolol ...")
        print(result["output"])              # final answer
        print(result["intermediate_steps"])  # tool call trace
    """

    SYSTEM_PROMPT = """You are a clinical operations AI agent for Healthcare-App hospital support.
You have access to five tools for searching the internal clinical knowledge base and looking up live patient and system data.

TOOL SELECTION GUIDE:
- Query contains MRN-XXXXXX → call lookup_patient_record once to get patient context.
- Query mentions a bug ID (BUG-*) or error code (ERR_*, AUTH-*, SCH-*) → call check_system_status once.
- Conceptual question (no identifiers) → call search_kb_semantic.
- Query has exact codes/IDs → call search_kb_bm25.
- Query mixes concepts and identifiers → call search_kb_hybrid.

IMPORTANT RULES:
- Call each tool at most once per query. Never repeat a tool call with the same input.
- Once you have enough information from the tools, stop calling tools and provide your final answer immediately.
- Always cite which tool and document title produced each piece of information.
- Treat any patient MRN as PHI — do not repeat it unnecessarily."""

    def __init__(self, engine: "RetrievalEngine") -> None:
        """
        Bind the retrieval engine, assemble the five-tool set, and construct
        the ReAct AgentExecutor ready to handle queries.

        Args:
            engine: An initialised RetrievalEngine that provides the three
                    KB search modes used by the search_kb_* tools.
        """
        self._engine = engine
        self._tools = _make_kb_tools(engine) + _make_system_tools()
        self._executor = self._build_executor()

    def _build_executor(self) -> Any:
        """
        Build a LangGraph ReAct agent using langgraph.prebuilt.create_react_agent.
        Recursion is capped at 50 steps via config at invoke time.
        """
        try:
            from langgraph.prebuilt import create_react_agent
        except ImportError as exc:
            raise ImportError("pip install langgraph") from exc

        from langchain_core.messages import SystemMessage
        from track2_retrieval_engine import build_llm
        llm = build_llm()

        return create_react_agent(
            model=llm,
            tools=self._tools,
            prompt=SystemMessage(content=self.SYSTEM_PROMPT),
        )

    def run(self, query: str) -> dict:
        """
        Run the agent synchronously.
        Returns {"output": str, "intermediate_steps": list} where each step is
        a (action, observation) pair with action.tool and action.tool_input attrs.
        """
        from types import SimpleNamespace
        try:
            result = self._executor.invoke(
                {"messages": [("human", query)]},
                config={"recursion_limit": 50},
            )
            messages = result.get("messages", [])

            # Extract final answer from last AI message
            output = ""
            for msg in reversed(messages):
                if hasattr(msg, "content") and not hasattr(msg, "tool_call_id"):
                    if not getattr(msg, "tool_calls", None):
                        output = msg.content
                        break

            # Extract intermediate steps: (action, observation) pairs
            steps = []
            for i, msg in enumerate(messages):
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        obs = next(
                            (m.content for m in messages[i + 1:]
                             if getattr(m, "tool_call_id", None) == tc["id"]),
                            "",
                        )
                        action = SimpleNamespace(tool=tc["name"], tool_input=tc["args"])
                        steps.append((action, obs))

            return {"output": output, "intermediate_steps": steps}
        except Exception as exc:
            logger.error("Agent error: %s", exc)
            return {"output": f"Agent error: {exc}", "intermediate_steps": []}

    def run_stream(self, query: str) -> Iterator[str]:
        """Stream final-answer tokens for Streamlit chat_message."""
        try:
            for chunk in self._executor.stream(
                {"messages": [("human", query)]},
                config={"recursion_limit": 50},
                stream_mode="values",
            ):
                messages = chunk.get("messages", [])
                if messages:
                    last = messages[-1]
                    if hasattr(last, "content") and not getattr(last, "tool_calls", None) \
                            and not hasattr(last, "tool_call_id"):
                        yield last.content
        except Exception as exc:
            yield f"\n[Agent error: {exc}]"


# ── Demo ──────────────────────────────────────────────────────────────────────

def _print_case(agent: ClinicalAgent, label: str, query: str) -> None:
    """Print a formatted trace of tool calls and the final answer for one demo test case."""
    print(f"\n{'='*64}")
    print(f"Case {label}")
    print(f"Query: {query}")
    print("=" * 64)

    result = agent.run(query)

    print("\n🔧 Tool calls:")
    for step in result["intermediate_steps"]:
        action, observation = step
        obs_preview = str(observation)[:200].replace("\n", " ")
        print(f"  Tool: {action.tool}")
        print(f"  Input: {str(action.tool_input)[:120]}")
        print(f"  Output: {obs_preview}...")

    print(f"\n💬 Final answer:\n{result['output']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.path.insert(0, str(Path(__file__).parent))

    from track2_retrieval_engine import RetrievalEngine
    KB = Path(__file__).parent / "healthcare_app_knowledge_base.json"
    engine = RetrievalEngine(KB)
    agent = ClinicalAgent(engine)

    _print_case(
        agent,
        "1 — Identifier only",
        "Is BUG-EHR-2301 still active and what's the workaround?",
    )

    _print_case(
        agent,
        "2 — Conceptual multi-step",
        "A patient discharged yesterday can't book a follow-up, getting a conflict error — "
        "what's happening and when will it be fixed?",
    )

    _print_case(
        agent,
        "3 — Full pipeline (all 5 tools, qualifying scenario)",
        "MRN-334521 has two metoprolol 25mg orders at 08:00. Patient hasn't received either "
        "dose yet. What's the immediate step and is BUG-EHR-2301 still active?",
    )
