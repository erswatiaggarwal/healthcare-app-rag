"""
demo_runner.py

Interactive terminal demo for the Healthcare-App Clinical Knowledge Assistant.
Run:  python3 demo_runner.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

load_dotenv()
console = Console()

KB_PATH = Path(__file__).parent / "healthcare_app_knowledge_base.json"

DEMO_QUERIES = [
    ("Exact identifier",  "ERR_HL7_ROUTE_FAIL lab result not arriving in chart",        "bm25"),
    ("Conceptual",        "patient can't book follow-up after discharge",                "semantic"),
    ("Mixed (MRN + bug)", "MRN-334521 duplicate metoprolol 25mg orders BUG-EHR-2301",  "hybrid"),
]

AGENT_CASES = [
    ("Identifier only",
     "Is BUG-EHR-2301 still active and what is the workaround?"),
    ("Conceptual multi-step",
     "A patient discharged yesterday can't book a follow-up. Getting a conflict error — what's happening?"),
    ("Full pipeline (all 5 tools)",
     "MRN-334521 has two metoprolol 25mg orders at 08:00. Patient hasn't received either dose. What's the immediate step and is BUG-EHR-2301 still active?"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def pause(secs: float = 0.6) -> None:
    time.sleep(secs)


def section(title: str) -> None:
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan"))
    console.print()


def has_llm_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("NEBIUS_API_KEY"))


def llm_backend() -> str:
    if os.getenv("NEBIUS_API_KEY"):
        return "[green]Nebius AI Studio (PHI-safe)[/green]"
    if os.getenv("OPENAI_API_KEY"):
        return "[yellow]OpenAI Cloud (dev)[/yellow]"
    return "[red]No LLM key — agent demo skipped[/red]"


# ── Sections ──────────────────────────────────────────────────────────────────

def demo_banner() -> None:
    banner = Text(justify="center")
    banner.append("\n  Healthcare-App\n", style="bold white on #0d6e8a")
    banner.append("  Clinical Knowledge Assistant  \n", style="bold white on #0d6e8a")
    banner.append("  Track 2 — Hybrid RAG + ReAct Agent  \n", style="italic #b0d8e8 on #0d6e8a")
    console.print(Panel(banner, border_style="cyan", padding=(0, 4)))
    console.print()

    info_table = Table.grid(padding=(0, 4))
    info_table.add_column(style="bold cyan", no_wrap=True)
    info_table.add_column()
    info_table.add_row("Vector store",  "[blue]ChromaDB[/blue] (Pinecone if keys set)")
    info_table.add_row("Embeddings",    "[blue]all-MiniLM-L6-v2[/blue] (384-dim cosine)")
    info_table.add_row("LLM backend",   llm_backend())
    info_table.add_row("Retrieval",     "[blue]Semantic · BM25 · Hybrid RRF[/blue]")
    console.print(Align.center(info_table))
    pause()


def demo_kb_stats() -> None:
    section("Section 1 — Knowledge Base")

    docs = json.loads(KB_PATH.read_text())
    from collections import Counter
    by_type = Counter(d["doc_type"] for d in docs)
    by_area = Counter(d["clinical_area"] for d in docs)
    by_pri  = Counter(d["priority"] for d in docs)

    # KPI cards as columns
    cards = []
    for label, value, color in [
        ("Total Documents", str(len(docs)),                         "cyan"),
        ("Active Issues",   str(sum(1 for d in docs if d["status"] == "active")), "red"),
        ("Clinical Areas",  str(len(by_area)),                      "green"),
        ("Doc Types",       str(len(by_type)),                      "yellow"),
    ]:
        cards.append(Panel(
            f"[bold {color}]{value}[/bold {color}]\n[dim]{label}[/dim]",
            border_style=color, padding=(1, 4),
        ))
    console.print(Columns(cards))
    pause(0.4)

    # Breakdown table
    t = Table("Doc Type", "Count", "█", box=box.SIMPLE_HEAVY, header_style="bold cyan")
    max_c = max(by_type.values())
    for dt, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt / max_c * 20)
        t.add_row(dt, str(cnt), f"[cyan]{bar}[/cyan]")
    console.print(t)
    pause()


def demo_retrieval(engine: object) -> None:
    section("Section 2 — Retrieval Comparison (Semantic · BM25 · Hybrid)")

    for label, query, best_mode in DEMO_QUERIES:
        console.print(f"\n[bold yellow]▶ {label}[/bold yellow]")
        console.print(f"  [dim]Query:[/dim] {query}\n")

        t = Table("Mode", "Rank", "Score", "Title", "Retriever",
                  box=box.ROUNDED, header_style="bold white on #0d6e8a",
                  show_lines=False)

        for mode_key, mode_label, color, fn in [
            ("semantic", "🔍 Semantic",    "blue",   lambda q: engine.search_semantic(q, k=3)),
            ("bm25",     "📝 BM25",        "yellow", lambda q: engine.search_bm25(q, k=3)),
            ("hybrid",   "🔀 Hybrid RRF",  "cyan",   lambda q: engine.search_hybrid(q, k=3)),
        ]:
            results = fn(query)
            for rank, r in enumerate(results, 1):
                score_str = f"{r.score:.4f}" if r.score is not None else "—"
                retriever = r.retriever if "hybrid" in r.retriever else ""
                star = " ⭐" if mode_key == best_mode and rank == 1 else ""
                t.add_row(
                    f"[{color}]{mode_label}[/{color}]" if rank == 1 else "",
                    str(rank),
                    f"[{color}]{score_str}[/{color}]",
                    r.title[:50] + star,
                    f"[dim]{retriever}[/dim]",
                )

        console.print(t)
        pause(0.5)

    console.print()
    console.print(Panel(
        "[bold]Key insight:[/bold]\n"
        "• [yellow]BM25[/yellow] dominates on exact identifiers (ERR_*, MRN-*, BUG-*)\n"
        "• [blue]Semantic[/blue] bridges vocabulary gaps (\"conflict error\" → SCH-CONFLICT-88)\n"
        "• [cyan]Hybrid RRF[/cyan] gets the best of both — tagged [bold]hybrid-both[/bold] = highest confidence",
        border_style="cyan", title="📖 Retrieval Insight",
    ))
    pause()


def demo_rrf_deep_dive(engine: object) -> None:
    section("Section 3 — RRF Deep Dive")

    query = DEMO_QUERIES[2][1]  # mixed query
    console.print(f"[dim]Query:[/dim] [bold]{query}[/bold]\n")

    results = engine.search_hybrid(query, k=5)
    t = Table("RRF Score", "Tag", "v-rank", "b-rank", "Title",
              box=box.SIMPLE_HEAD, header_style="bold cyan")
    tag_color = {"hybrid-both": "green", "hybrid-semantic": "blue", "hybrid-bm25": "yellow"}
    for r in results:
        color = tag_color.get(r.retriever, "white")
        t.add_row(
            f"[cyan]{r.score:.6f}[/cyan]",
            f"[{color}]{r.retriever}[/{color}]",
            str(r.vector_rank) if r.vector_rank is not None else "—",
            str(r.bm25_rank)   if r.bm25_rank   is not None else "—",
            r.title[:55],
        )
    console.print(t)
    console.print()
    console.print("[dim]Formula: score = 0.6/(60+vrank) + 0.4/(60+brank)[/dim]")
    pause()


def demo_agent() -> None:
    section("Section 4 — ReAct Agent (5 Tools)")

    if not has_llm_key():
        console.print(Panel(
            "[yellow]No LLM API key found.[/yellow]\n"
            "Set [bold]OPENAI_API_KEY[/bold] or [bold]NEBIUS_API_KEY[/bold] in .env to run agent cases.\n"
            "Skipping Section 4.",
            border_style="yellow", title="⚠️  Skipped",
        ))
        return

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as prog:
        task = prog.add_task("Loading engine and agent…", total=None)
        from track2_retrieval_engine import RetrievalEngine
        from track2_clinical_agent import ClinicalAgent
        engine = RetrievalEngine(KB_PATH)
        agent  = ClinicalAgent(engine)
        prog.update(task, description="Ready ✓")

    tool_icon = {
        "search_kb_semantic":    "🔍",
        "search_kb_bm25":        "📝",
        "search_kb_hybrid":      "🔀",
        "lookup_patient_record": "👤",
        "check_system_status":   "🔧",
    }

    for label, query in AGENT_CASES:
        console.print(f"\n[bold yellow]▶ Case: {label}[/bold yellow]")
        console.print(Panel(query, border_style="dim", padding=(0, 2)))

        with Progress(SpinnerColumn(), TextColumn("Agent reasoning…"), transient=True) as prog:
            prog.add_task("", total=None)
            result = agent.run(query)

        steps = result.get("intermediate_steps", [])
        output = result.get("output", "")

        # Tool call timeline
        if steps:
            console.print(f"\n[bold]Tool calls ({len(steps)}):[/bold]")
            for i, (action, obs) in enumerate(steps, 1):
                icon = tool_icon.get(action.tool, "🔹")
                inp  = str(action.tool_input)[:80]
                obs_preview = str(obs)[:120].replace("\n", " ")
                console.print(f"  {i}. {icon} [cyan]{action.tool}[/cyan]([dim]{inp}[/dim])")
                console.print(f"     [dim]↳ {obs_preview}…[/dim]")

        # Final answer
        console.print()
        console.print(Panel(
            output,
            title="[bold green]Final Answer[/bold green]",
            border_style="green",
            padding=(0, 2),
        ))
        pause(0.8)


def demo_summary() -> None:
    section("Summary")

    rows = [
        ("healthcare_app_knowledge_base.json", "50-doc clinical KB",                  "✅"),
        ("track2_retrieval_engine.py",         "Semantic · BM25 · Hybrid RRF",        "✅"),
        ("track2_pdf_ingestor.py",             "PyMuPDF ingestion + BM25 rebuild",    "✅"),
        ("track2_clinical_agent.py",           "ReAct agent · 5 tools · LangGraph",   "✅"),
        ("track2_kb_viewer.py",                "Streamlit app · 5 tabs",              "✅"),
        ("track2_hybrid_rag.ipynb",            "Teaching notebook · Sections 0–7",    "✅"),
    ]

    t = Table("File", "What it does", "Status",
              box=box.ROUNDED, header_style="bold white on #0d6e8a")
    for file, desc, status in rows:
        t.add_row(f"[cyan]{file}[/cyan]", desc, f"[green]{status}[/green]")
    console.print(t)
    console.print()

    console.print(Panel(
        "[bold]To launch the full UI:[/bold]\n\n"
        "  [cyan]streamlit run track2_kb_viewer.py[/cyan]\n\n"
        "[bold]To run the teaching notebook:[/bold]\n\n"
        "  [cyan]jupyter notebook track2_hybrid_rag.ipynb[/cyan]\n\n"
        f"[bold]LLM backend:[/bold] {llm_backend()}",
        border_style="cyan", title="🚀 Next Steps",
    ))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    console.clear()
    demo_banner()

    # Section 1 — KB stats (no LLM needed)
    demo_kb_stats()

    # Section 2 & 3 — Retrieval (no LLM needed)
    with Progress(SpinnerColumn(), TextColumn("Initialising retrieval engine…"), transient=True) as prog:
        prog.add_task("", total=None)
        from track2_retrieval_engine import RetrievalEngine
        engine = RetrievalEngine(KB_PATH)

    demo_retrieval(engine)
    demo_rrf_deep_dive(engine)

    # Section 4 — Agent (LLM needed)
    demo_agent()

    # Summary
    demo_summary()
    console.print()


if __name__ == "__main__":
    main()
