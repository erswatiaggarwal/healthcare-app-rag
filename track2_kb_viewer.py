"""
track2_kb_viewer.py

Streamlit Clinical Knowledge Base viewer — Healthcare-App.
Tabs: Dashboard | KB Browser | Search Lab | AI Assistant | AI Agent | PDF Ingestor
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# On Streamlit Community Cloud, secrets live in st.secrets (not .env).
# Copy them into os.environ so every os.getenv() call in this file and
# in track2_retrieval_engine / track2_clinical_agent works unchanged.
for _k, _v in st.secrets.items():
    if isinstance(_v, str):
        os.environ.setdefault(_k, _v)

# ── Constants ─────────────────────────────────────────────────────────────────

KB_PATH    = Path(__file__).parent / "healthcare_app_knowledge_base.json"
NAMESPACE  = "healthcare-app-rag"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS  = 512

ACCENT       = "#00b4d8"
ACCENT_DARK  = "#0096c7"
ACCENT_LIGHT = "#0a2a35"
SUCCESS      = "#22c55e"
WARNING_COL  = "#f59e0b"
DANGER       = "#ef4444"
SURFACE      = "#0f0f0f"
CARD_BG      = "#1a1a1a"

CLINICAL_AREA_ICON: dict[str, str] = {
    "admissions":  "🏥", "medications": "💊", "lab_results": "🧪",
    "discharge":   "🚪", "scheduling":  "📅", "telehealth":  "📹",
    "insurance":   "📋", "account":     "👤", "general":     "📄",
}
PRIORITY_COLOR: dict[str, str] = {
    "P0": DANGER, "P1": "#e74c3c", "P2": WARNING_COL, "P3": "#64748b",
}
PRIORITY_LABEL: dict[str, str] = {
    "P0": "🔴 P0 Critical", "P1": "🔴 P1 High",
    "P2": "🟠 P2 Medium",   "P3": "⚪ P3 Low",
}
RAG_STRATEGIES: dict[str, dict] = {
    "hybrid":   {"label": "🔀 Hybrid (RRF)", "help": "Best overall — fuses semantic + BM25."},
    "semantic": {"label": "🔍 Semantic",      "help": "Conceptual questions, paraphrase matching."},
    "bm25":     {"label": "📝 BM25 Keyword",  "help": "Exact IDs, error codes, MRN numbers."},
}
RAG_PROMPT = """You are a clinical operations assistant for Healthcare-App hospital support.
Use the context below to answer concisely. State the likely cause and the immediate next step.
Cite document titles when relevant.

Context:
{context}

Clinical issue: {question}
Support guidance:"""

MRN_PATTERN = re.compile(r"MRN-\d{6}")

TOOL_ICON: dict[str, str] = {
    "search_kb_semantic":    "🔍",
    "search_kb_bm25":        "📝",
    "search_kb_hybrid":      "🔀",
    "lookup_patient_record": "👤",
    "check_system_status":   "🔧",
}
TOOL_COLOR: dict[str, str] = {
    "search_kb_semantic":    "#3b82f6",
    "search_kb_bm25":        "#f59e0b",
    "search_kb_hybrid":      "#0d6e8a",
    "lookup_patient_record": "#8b5cf6",
    "check_system_status":   "#10b981",
}


# ── CSS ───────────────────────────────────────────────────────────────────────

def inject_css() -> None:
    """Inject global custom CSS: hero header, metric cards, doc cards, tool timeline."""
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

      /* ═══════════════════════════════════════════════════════════
         GLOBAL — black background + light text on every surface
         ═══════════════════════════════════════════════════════════ */
      html, body {{ background: #000 !important; color: #e2e8f0 !important; }}
      html, body, [class*="css"], * {{ font-family: 'Inter', sans-serif; }}

      .stApp,
      .stApp > div,
      section[data-testid="stAppViewContainer"],
      .main, .main > div,
      .block-container,
      [data-testid="stVerticalBlock"],
      [data-testid="stVerticalBlockBorderWrapper"],
      [data-testid="column"],
      [data-testid="stForm"] {{
        background-color: #000000 !important;
        color: #e2e8f0 !important;
      }}

      /* All plain text */
      p, li, td, th, pre, blockquote {{ color: #e2e8f0 !important; }}
      h1, h2, h3, h4, h5, h6 {{ color: #f1f5f9 !important; }}
      a {{ color: {ACCENT} !important; }}
      strong, b {{ color: #f1f5f9 !important; }}
      em, i {{ color: #cbd5e1 !important; }}

      #MainMenu, footer {{ visibility: hidden; }}
      /* keep header visible so the Deploy button shows */
      header {{ visibility: visible !important; background: #0a0a0a !important; border-bottom: 1px solid #2d2d2d !important; }}
      header * {{ color: #e2e8f0 !important; }}
      header button {{ background: transparent !important; color: #e2e8f0 !important; }}
      /* style the Deploy button specifically */
      [data-testid="stToolbar"] {{ background: transparent !important; }}
      [data-testid="stDeployButton"] button {{
        background: {ACCENT} !important;
        color: #000 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        border: none !important;
      }}
      .block-container {{ padding-top: 0.5rem !important; }}

      /* ═══════════════════════════════════════════════════════════
         FORM WIDGETS
         ═══════════════════════════════════════════════════════════ */

      /* labels */
      label, .stTextInput label, .stSelectbox label, .stSlider label,
      .stRadio label, .stCheckbox label, .stFileUploader label,
      .stNumberInput label, .stDateInput label, .stTimeInput label,
      .stMultiSelect label, .stTextArea label,
      [data-testid="stWidgetLabel"] p,
      [data-testid="stWidgetLabel"] span {{
        color: #94a3b8 !important;
      }}

      /* text inputs */
      .stTextInput input,
      .stNumberInput input,
      .stTextArea textarea,
      input[type="text"], input[type="number"], input[type="email"],
      textarea {{
        background: #1a1a1a !important;
        color: #e2e8f0 !important;
        border: 1px solid #2d2d2d !important;
        border-radius: 6px !important;
      }}
      .stTextInput input:focus, .stNumberInput input:focus,
      .stTextArea textarea:focus {{
        border-color: {ACCENT} !important;
        box-shadow: 0 0 0 2px rgba(0,180,216,0.2) !important;
      }}

      /* selectbox / multiselect trigger */
      div[data-baseweb="select"] > div,
      div[data-baseweb="select"] > div > div,
      div[data-baseweb="select"] span {{
        background: #1a1a1a !important;
        border-color: #2d2d2d !important;
        color: #e2e8f0 !important;
      }}
      /* dropdown option list */
      ul[data-baseweb="menu"],
      li[role="option"],
      div[data-baseweb="popover"],
      div[data-baseweb="popover"] * {{
        background: #1a1a1a !important;
        color: #e2e8f0 !important;
      }}
      li[role="option"]:hover,
      li[aria-selected="true"] {{
        background: #0a2a35 !important;
        color: {ACCENT} !important;
      }}

      /* radio */
      .stRadio > div > label > div:first-child {{
        border-color: #2d2d2d !important;
      }}
      .stRadio > div > label > div:first-child[aria-checked="true"] {{
        background: {ACCENT} !important;
        border-color: {ACCENT} !important;
      }}

      /* checkbox */
      [data-baseweb="checkbox"] > div {{
        background: #1a1a1a !important;
        border-color: #2d2d2d !important;
      }}

      /* slider */
      [data-testid="stSlider"] [data-testid="stSliderTrack"] {{
        background: #2d2d2d !important;
      }}

      /* file uploader */
      [data-testid="stFileUploader"] section {{
        background: #111111 !important;
        border: 2px dashed #2d2d2d !important;
        color: #94a3b8 !important;
      }}
      [data-testid="stFileUploader"] section * {{
        color: #94a3b8 !important;
      }}

      /* form container */
      [data-testid="stForm"] {{
        border: 1px solid #2d2d2d !important;
        border-radius: 8px !important;
        background: #0a0a0a !important;
      }}

      /* ═══════════════════════════════════════════════════════════
         EXPANDERS
         ═══════════════════════════════════════════════════════════ */
      .streamlit-expanderHeader,
      [data-testid="stExpander"] summary,
      details summary {{
        background: #1a1a1a !important;
        color: #e2e8f0 !important;
        border: 1px solid #2d2d2d !important;
        border-radius: 6px !important;
      }}
      .streamlit-expanderContent,
      [data-testid="stExpander"] > div[data-testid="stExpanderDetails"],
      details > div {{
        background: #111111 !important;
        border: 1px solid #2d2d2d !important;
        border-top: none !important;
        color: #e2e8f0 !important;
      }}
      [data-testid="stExpander"] *,
      .streamlit-expanderContent * {{
        color: #e2e8f0 !important;
      }}

      /* ═══════════════════════════════════════════════════════════
         ALERT / INFO / WARNING / SUCCESS / ERROR BOXES
         ═══════════════════════════════════════════════════════════ */
      .stAlert,
      [data-testid="stAlert"],
      div[role="alert"],
      .stException {{
        background: #111111 !important;
        border-color: #2d2d2d !important;
        color: #e2e8f0 !important;
      }}
      .stAlert p, .stAlert span, .stAlert div,
      [data-testid="stAlert"] p,
      [data-testid="stAlert"] span,
      [data-testid="stAlert"] div {{
        color: #e2e8f0 !important;
      }}
      /* info — blue tint border */
      [data-testid="stAlert"][data-baseweb="notification"][kind="info"] {{
        background: #0a1a2e !important;
        border-color: #1e3a5f !important;
      }}
      /* warning — amber tint border */
      [data-testid="stAlert"][kind="warning"] {{
        background: #1a1200 !important;
        border-color: #4a3000 !important;
      }}
      /* error — red tint border */
      [data-testid="stAlert"][kind="error"] {{
        background: #1a0505 !important;
        border-color: #7f1d1d !important;
      }}
      /* success — green tint border */
      [data-testid="stAlert"][kind="success"] {{
        background: #021a0e !important;
        border-color: #14532d !important;
      }}

      /* ═══════════════════════════════════════════════════════════
         TABS
         ═══════════════════════════════════════════════════════════ */
      .stTabs [data-baseweb="tab-list"] {{
        gap: 2px; background: #111111;
        border: 1px solid #2d2d2d; border-radius: 8px; padding: 4px;
      }}
      .stTabs [data-baseweb="tab"] {{
        border-radius: 6px; padding: 8px 16px;
        font-size: 13px; font-weight: 500; color: #64748b; background: transparent;
      }}
      .stTabs [aria-selected="true"] {{
        background: {ACCENT} !important; color: #000 !important;
        box-shadow: 0 0 12px rgba(0,180,216,0.4); font-weight: 600 !important;
      }}
      /* tab panel background */
      [data-testid="stTabPanel"] {{
        background: #000 !important;
      }}

      /* ═══════════════════════════════════════════════════════════
         BUTTONS
         ═══════════════════════════════════════════════════════════ */
      .stButton > button {{
        background: {ACCENT}; color: #000; border: none;
        border-radius: 7px; font-weight: 600; padding: 8px 20px;
        transition: background 0.15s, box-shadow 0.15s;
      }}
      .stButton > button:hover {{
        background: #48cae4; box-shadow: 0 0 16px rgba(0,180,216,0.4);
      }}
      /* secondary / ghost buttons (used in KB list) */
      .stButton > button[kind="secondary"] {{
        background: #111111 !important; color: #e2e8f0 !important;
        border: 1px solid #2d2d2d !important;
      }}
      .stButton > button[kind="secondary"]:hover {{
        background: #1a1a1a !important; border-color: {ACCENT} !important;
        color: {ACCENT} !important; box-shadow: none !important;
      }}

      /* form submit button */
      .stFormSubmitButton > button {{
        background: {ACCENT}; color: #000; border: none;
        border-radius: 7px; font-weight: 600;
      }}

      /* ═══════════════════════════════════════════════════════════
         CHAT (Assistant + AI Agent tabs)
         ═══════════════════════════════════════════════════════════ */
      [data-testid="stChatMessage"] {{
        background: #111111 !important;
        border: 1px solid #2d2d2d !important;
        border-radius: 10px !important;
        margin-bottom: 8px !important;
      }}
      [data-testid="stChatMessage"] p,
      [data-testid="stChatMessage"] span,
      [data-testid="stChatMessage"] div,
      [data-testid="stChatMessage"] li,
      [data-testid="stChatMessage"] td,
      [data-testid="stChatMessage"] strong,
      [data-testid="stChatMessage"] em {{
        color: #e2e8f0 !important;
      }}
      /* user message — teal tint */
      [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
        background: #0a2a35 !important;
        border-color: {ACCENT} !important;
      }}
      /* chat input */
      [data-testid="stChatInput"] textarea {{
        background: #1a1a1a !important; color: #e2e8f0 !important;
        border: 1px solid #2d2d2d !important;
      }}
      [data-testid="stChatInput"] textarea:focus {{
        border-color: {ACCENT} !important;
      }}
      [data-testid="stChatInputSubmitButton"] {{ color: {ACCENT} !important; }}

      /* ═══════════════════════════════════════════════════════════
         SIDEBAR
         ═══════════════════════════════════════════════════════════ */
      [data-testid="stSidebar"] {{
        background: #0a0a0a !important; border-right: 1px solid #2d2d2d;
      }}
      [data-testid="stSidebar"] * {{ color: #e2e8f0 !important; }}
      [data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background: #1a1a1a !important;
      }}

      /* ═══════════════════════════════════════════════════════════
         PROGRESS / SPINNER
         ═══════════════════════════════════════════════════════════ */
      .stProgress > div > div {{ background: {ACCENT} !important; }}
      [data-testid="stProgressBar"] > div {{ background: {ACCENT} !important; }}

      /* ═══════════════════════════════════════════════════════════
         CHARTS
         ═══════════════════════════════════════════════════════════ */
      .vega-embed, .vega-embed * {{ background: transparent !important; }}
      canvas {{ background: transparent !important; }}

      /* ═══════════════════════════════════════════════════════════
         CODE / PRE
         ═══════════════════════════════════════════════════════════ */
      code {{ background: #1a1a1a !important; color: {ACCENT} !important; border-radius: 3px; }}
      pre  {{ background: #0f0f0f !important; border: 1px solid #2d2d2d !important;
              color: #e2e8f0 !important; }}

      /* ═══════════════════════════════════════════════════════════
         DIVIDER
         ═══════════════════════════════════════════════════════════ */
      hr {{ border-color: #2d2d2d !important; }}

      /* ═══════════════════════════════════════════════════════════
         CUSTOM COMPONENTS
         ═══════════════════════════════════════════════════════════ */

      /* Hero */
      .hero {{
        background: linear-gradient(135deg, #001a22 0%, #003344 50%, #004d66 100%);
        border: 1px solid {ACCENT}; border-radius: 12px; padding: 28px 36px;
        margin-bottom: 24px; color: white; display: flex;
        align-items: center; justify-content: space-between;
        box-shadow: 0 0 32px rgba(0,180,216,0.15);
      }}
      .hero-title {{ font-size: 26px; font-weight: 700; letter-spacing: -0.5px; margin: 0; color: white !important; }}
      .hero-sub   {{ font-size: 13px; opacity: 0.75; margin-top: 4px; color: #cce9f5 !important; }}
      .hero-badge {{
        background: rgba(0,180,216,0.15); border: 1px solid rgba(0,180,216,0.4);
        border-radius: 20px; padding: 6px 14px; font-size: 12px;
        font-weight: 500; color: {ACCENT} !important;
      }}

      /* Metric cards */
      .metric-card {{
        background: #111111; border: 1px solid #2d2d2d;
        border-top: 2px solid {ACCENT}; border-radius: 10px;
        padding: 18px 20px; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
        transition: transform 0.15s, box-shadow 0.15s;
      }}
      .metric-card:hover {{
        transform: translateY(-2px); box-shadow: 0 4px 20px rgba(0,180,216,0.2);
      }}
      .metric-value {{ font-size: 32px; font-weight: 700; color: {ACCENT} !important; line-height: 1; }}
      .metric-label {{ font-size: 12px; color: #64748b !important; margin-top: 4px; font-weight: 500; }}
      .metric-delta {{ font-size: 11px; margin-top: 6px; }}

      /* Badge + chip */
      .badge {{
        display: inline-block; padding: 2px 9px; border-radius: 999px;
        font-size: 11px; font-weight: 600; color: white !important; letter-spacing: 0.3px;
      }}
      .chip {{
        display: inline-block; background: #0a2a35; color: {ACCENT} !important;
        border: 1px solid #1a4a5e; padding: 2px 8px; border-radius: 999px;
        font-size: 11px; font-weight: 500; margin-right: 4px; margin-bottom: 2px;
      }}

      /* Result cards */
      .result-card {{
        background: #111111; border: 1px solid #2d2d2d; border-radius: 10px;
        padding: 14px 16px; margin-bottom: 10px; transition: border-color 0.15s;
      }}
      .result-card:hover {{ border-color: {ACCENT}; }}
      .result-score {{ font-size: 22px; font-weight: 700; color: {ACCENT} !important; }}
      .result-title {{ font-size: 14px; font-weight: 600; color: #f1f5f9 !important; margin-top: 4px; }}
      .result-snippet {{ font-size: 12px; color: #94a3b8 !important; margin-top: 6px; line-height: 1.5; }}

      /* Tool timeline */
      .tool-step {{ display: flex; align-items: flex-start; gap: 12px;
                    padding: 10px 0; border-bottom: 1px solid #1e1e1e; }}
      .tool-icon {{ width: 36px; height: 36px; border-radius: 50%;
                    display: flex; align-items: center; justify-content: center;
                    font-size: 16px; flex-shrink: 0; color: white; }}
      .tool-name  {{ font-size: 13px; font-weight: 600; color: #f1f5f9 !important; }}
      .tool-input {{ font-size: 11px; color: #64748b !important; font-family: monospace; margin-top: 2px; }}
      .tool-obs   {{ font-size: 11px; color: #94a3b8 !important; margin-top: 4px;
                     background: #0f0f0f; border: 1px solid #2d2d2d;
                     border-radius: 4px; padding: 4px 8px; }}

      /* Chat sources pill */
      .chat-sources {{
        background: #0a2a35; border: 1px solid #1a4a5e; border-radius: 8px;
        padding: 8px 12px; margin-top: 8px; font-size: 11px; color: {ACCENT} !important;
      }}

      /* Filter bar */
      .filter-bar {{
        background: #111111; border: 1px solid #2d2d2d;
        border-radius: 10px; padding: 14px 16px; margin-bottom: 14px;
      }}

      /* PHI banner */
      .phi-banner {{
        background: #1a0505; border: 1px solid #7f1d1d;
        border-left: 4px solid {DANGER}; border-radius: 8px;
        padding: 10px 14px; font-size: 13px; color: #fca5a5 !important; margin: 8px 0;
      }}
    </style>
    """, unsafe_allow_html=True)


# ── Data & resource loading ───────────────────────────────────────────────────

@st.cache_data
def load_docs() -> list[dict]:
    """Load and cache all 50 knowledge base documents from the JSON file."""
    return json.loads(KB_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def get_engine() -> Any:
    """Instantiate and cache the RetrievalEngine (vector store init and BM25 build run once)."""
    from track2_retrieval_engine import RetrievalEngine
    return RetrievalEngine(KB_PATH)


@st.cache_resource
def get_agent() -> Any:
    """Instantiate and cache the ClinicalAgent. Returns None if initialisation fails."""
    try:
        from track2_clinical_agent import ClinicalAgent
        return ClinicalAgent(get_engine())
    except Exception as exc:
        st.error(f"Agent init failed: {exc}")
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def badge(text: str, color: str) -> str:
    """Return a styled HTML pill badge."""
    return f'<span class="badge" style="background:{color}">{text}</span>'


def chip(text: str) -> str:
    """Return a styled HTML metadata chip."""
    return f'<span class="chip">{text}</span>'


def _has_api_key() -> bool:
    """Return True if at least one LLM API key (OpenAI or Nebius) is configured."""
    return bool(os.getenv("OPENAI_API_KEY") or os.getenv("NEBIUS_API_KEY"))


def _nebius_active() -> bool:
    """Return True when NEBIUS_API_KEY is set — PHI-safe Nebius backend is active."""
    return bool(os.getenv("NEBIUS_API_KEY"))


def _backend_badge() -> str:
    """Return an HTML backend-status badge string."""
    if _nebius_active():
        return badge("🟢 Nebius AI Studio  PHI-safe", SUCCESS)
    if os.getenv("OPENAI_API_KEY"):
        return badge("🟡 OpenAI Cloud  dev only", WARNING_COL)
    return badge("🔴 No API key", DANGER)


# ── Hero header ───────────────────────────────────────────────────────────────

def render_hero(docs: list[dict]) -> None:
    """Render the gradient hero banner with live backend status badge."""
    store = "Pinecone" if os.getenv("PINECONE_API_KEY") else "ChromaDB"
    st.markdown(f"""
    <div class="hero">
      <div>
        <div class="hero-title">🏥 Healthcare-App Clinical Knowledge Assistant</div>
        <div class="hero-sub">Hybrid RAG · BM25 + Semantic + ReAct Agent · {len(docs)} KB documents</div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <span class="hero-badge">🗄 {store}</span>
        <span class="hero-badge">🤖 ReAct Agent</span>
        {"<span class='hero-badge'>🔒 PHI-safe</span>" if _nebius_active() else ""}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Tab 1: Dashboard ──────────────────────────────────────────────────────────

def tab_dashboard(docs: list[dict]) -> None:
    """
    Render Tab 1 — Dashboard.
    Shows KPI metric cards, document-type and clinical-area distribution charts,
    and a live backend status panel.
    """
    by_type  = Counter(d["doc_type"]      for d in docs)
    by_area  = Counter(d["clinical_area"] for d in docs)
    by_pri   = Counter(d["priority"]      for d in docs)
    active   = sum(1 for d in docs if d["status"] == "active")

    # ── KPI row ──────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, value, label, delta in [
        (c1, len(docs),         "Total Documents",   None),
        (c2, active,            "Active Issues",     f"{active} need attention"),
        (c3, len(by_area),      "Clinical Areas",    None),
        (c4, len(by_type),      "Document Types",    None),
        (c5, by_pri.get("P0",0)+by_pri.get("P1",0), "P0/P1 Critical", None),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{value}</div>
              <div class="metric-label">{label}</div>
              {"<div class='metric-delta' style='color:#ef4444'>⚠ "+delta+"</div>" if delta else ""}
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📂 Documents by Type")
        try:
            import altair as alt, pandas as pd
            df = pd.DataFrame({"Type": list(by_type.keys()), "Count": list(by_type.values())})
            chart = alt.Chart(df).mark_bar(
                cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=ACCENT,
            ).encode(
                x=alt.X("Count:Q", title=""),
                y=alt.Y("Type:N",  sort="-x", title=""),
                tooltip=["Type", "Count"],
                color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None),
            ).properties(height=180)
            st.altair_chart(chart, use_container_width=True)
        except ImportError:
            for dt, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
                st.markdown(f"`{dt}` — **{cnt}**")

    with col_b:
        st.markdown("#### 🏥 Documents by Clinical Area")
        try:
            import altair as alt, pandas as pd
            df2 = pd.DataFrame({"Area": [CLINICAL_AREA_ICON.get(k,"📄")+" "+k for k in by_area],
                                 "Count": list(by_area.values())})
            c2 = alt.Chart(df2).mark_bar(
                cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
            ).encode(
                x=alt.X("Count:Q", title=""),
                y=alt.Y("Area:N",  sort="-x", title=""),
                tooltip=["Area", "Count"],
                color=alt.Color("Count:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
            ).properties(height=180)
            st.altair_chart(c2, use_container_width=True)
        except ImportError:
            for area, cnt in sorted(by_area.items(), key=lambda x: -x[1]):
                st.markdown(f"{CLINICAL_AREA_ICON.get(area,'📄')} `{area}` — **{cnt}**")

    # ── Status & quick links ──────────────────────────────────────
    st.markdown("---")
    col_s, col_q = st.columns([1, 1])

    with col_s:
        st.markdown("#### ⚙️ System Status")
        store = "Pinecone ☁️" if os.getenv("PINECONE_API_KEY") else "ChromaDB 💾 (local)"
        rows = [
            ("Vector Store",  store,                                               "🟢"),
            ("LLM Backend",   "Nebius AI Studio" if _nebius_active() else "OpenAI Cloud",
             "🟢" if _has_api_key() else "🔴"),
            ("PHI-safe mode", "Active" if _nebius_active() else "Inactive",
             "🟢" if _nebius_active() else "🟡"),
            ("KB Documents",  f"{len(docs)} loaded",                              "🟢"),
        ]
        for label, value, status in rows:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:6px 0;border-bottom:1px solid #1e1e1e;font-size:13px'>"
                f"<span style='color:#64748b'>{label}</span>"
                f"<span style='font-weight:500'>{status} {value}</span></div>",
                unsafe_allow_html=True,
            )

    with col_q:
        st.markdown("#### 💡 Quick Queries to Try")
        examples = [
            ("🔴", "MRN-334521 duplicate metoprolol 25mg orders"),
            ("🧪", "ERR_HL7_ROUTE_FAIL lab result not arriving"),
            ("📅", "patient can't book follow-up after discharge"),
            ("🔧", "BUG-EHR-2301 — is it still active?"),
            ("💊", "LTC admission medication sync failure"),
        ]
        for icon, ex in examples:
            st.markdown(
                f"<div style='background:#111111;border-radius:6px;padding:6px 10px;"
                f"margin-bottom:5px;font-size:12px;cursor:pointer;border:1px solid #2d2d2d;color:#e2e8f0'>"
                f"{icon} {ex}</div>",
                unsafe_allow_html=True,
            )


# ── Tab 2: KB Browser ─────────────────────────────────────────────────────────

def tab_kb(docs: list[dict]) -> None:
    """
    Render Tab 2 — Knowledge Base Browser.
    Inline filter bar + text search. Master/detail layout with styled doc cards.
    """
    # ── Inline filter bar ─────────────────────────────────────────
    st.markdown("<div class='filter-bar'>", unsafe_allow_html=True)
    search_q = st.text_input("", placeholder="🔍  Search titles and content…", key="kb_search")
    fc1, fc2, fc3, fc4, fc5 = st.columns([2, 2, 1, 1, 1])
    with fc1:
        sel_type = st.selectbox("Doc type",      ["All"] + sorted({d["doc_type"]      for d in docs}), key="kb_type")
    with fc2:
        sel_area = st.selectbox("Clinical area", ["All"] + sorted({d["clinical_area"] for d in docs}), key="kb_area")
    with fc3:
        sel_pri  = st.selectbox("Priority",      ["All", "P0", "P1", "P2", "P3"], key="kb_pri")
    with fc4:
        sel_stat = st.selectbox("Status",        ["All"] + sorted({d["status"]        for d in docs}), key="kb_stat")
    with fc5:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↺ Reset", use_container_width=True, key="kb_reset"):
            for k in ("kb_search", "kb_type", "kb_area", "kb_pri", "kb_stat"):
                st.session_state.pop(k, None)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    filtered = [
        d for d in docs
        if (sel_type == "All" or d["doc_type"]      == sel_type)
        and (sel_area == "All" or d["clinical_area"] == sel_area)
        and (sel_pri  == "All" or d["priority"]      == sel_pri)
        and (sel_stat == "All" or d["status"]        == sel_stat)
        and (not search_q or search_q.lower() in d["title"].lower()
             or search_q.lower() in d["content"].lower())
    ]
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    filtered.sort(key=lambda d: (priority_order.get(d["priority"], 4), d["title"]))

    st.markdown(
        f"<div style='font-size:12px;color:#94a3b8;margin-bottom:12px'>"
        f"Showing <b>{len(filtered)}</b> of {len(docs)} documents</div>",
        unsafe_allow_html=True,
    )

    col_list, col_detail = st.columns([1, 2])
    selected_idx = st.session_state.get("selected_doc_idx", 0)

    with col_list:
        for i, doc in enumerate(filtered[:50]):
            icon  = CLINICAL_AREA_ICON.get(doc["clinical_area"], "📄")
            color = PRIORITY_COLOR.get(doc["priority"], "#64748b")
            if st.button(
                f"{icon} {doc['title'][:48]}",
                key=f"doc_{i}",
                use_container_width=True,
            ):
                st.session_state["selected_doc_idx"] = i
                st.rerun()

    with col_detail:
        if filtered:
            idx = min(selected_idx, len(filtered) - 1)
            doc = filtered[idx]
            color = PRIORITY_COLOR.get(doc["priority"], "#64748b")

            st.markdown(
                badge(doc["priority"], color) + "  " +
                "".join(chip(t) for t in [
                    doc["doc_type"], doc["clinical_area"],
                    doc["platform"], doc["patient_tier"], doc["status"],
                ]),
                unsafe_allow_html=True,
            )
            st.markdown(f"### {CLINICAL_AREA_ICON.get(doc['clinical_area'],'📄')}  {doc['title']}")

            if doc.get("source_file"):
                st.info(f"📄 Source: {doc['source_file']}  ·  page {doc.get('page_number','?')}")

            st.markdown(
                f"<div style='background:#111111;border-radius:8px;padding:16px;"
                f"border:1px solid #2d2d2d;font-size:13.5px;line-height:1.7;color:#e2e8f0'>"
                f"{doc['content']}</div>",
                unsafe_allow_html=True,
            )


# ── Tab 3: Search Lab ─────────────────────────────────────────────────────────

def tab_search() -> None:
    """
    Render Tab 3 — Search Lab.
    Runs all three retrieval modes simultaneously for side-by-side comparison,
    with an Altair score chart and expandable result cards.
    """
    st.markdown("#### 🔬 Search Lab — Compare All Three Retrieval Modes")

    with st.form("search_form"):
        query = st.text_input("", placeholder="Enter a clinical query…", label_visibility="collapsed")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            mode = st.radio("Mode", list(RAG_STRATEGIES.keys()),
                            format_func=lambda k: RAG_STRATEGIES[k]["label"],
                            horizontal=True)
        with col2:
            k = st.slider("Top-k", 1, 10, 5)
        with col3:
            compare_all = st.checkbox("Compare all 3 modes", value=False)
        submitted = st.form_submit_button("🔬 Search", use_container_width=True)

    if not submitted or not query.strip():
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;color:#94a3b8'>
          <div style='font-size:40px'>🔍</div>
          <div style='font-size:15px;margin-top:8px'>Enter a query above to search the knowledge base</div>
          <div style='font-size:12px;margin-top:6px'>Try: <code>ERR_HL7_ROUTE_FAIL</code>
            · <code>patient can't book follow-up</code>
            · <code>MRN-334521 metoprolol</code></div>
        </div>
        """, unsafe_allow_html=True)
        return

    engine = get_engine()

    if compare_all:
        # ── Side-by-side 3-mode comparison ────────────────────────
        with st.spinner("Running all 3 retrieval modes…"):
            sem = engine.search_semantic(query, k=k)
            bm  = engine.search_bm25(query, k=k)
            hyb = engine.search_hybrid(query, k=k)

        try:
            import altair as alt, pandas as pd
            rows = []
            for mode_name, results in [("Semantic", sem), ("BM25", bm), ("Hybrid", hyb)]:
                for r in results:
                    rows.append({"Mode": mode_name, "Title": r.title[:30], "Score": r.score or 0.0})
            df = pd.DataFrame(rows)
            color_scale = alt.Scale(domain=["Semantic","BM25","Hybrid"],
                                    range=["#3b82f6","#f59e0b",ACCENT])
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("Score:Q"),
                y=alt.Y("Title:N", sort="-x"),
                color=alt.Color("Mode:N", scale=color_scale),
                row=alt.Row("Mode:N", spacing=6),
                tooltip=["Mode","Title","Score"],
            ).properties(height=120)
            st.altair_chart(chart, use_container_width=True)
        except ImportError:
            pass

        cols = st.columns(3)
        for col, (mode_name, results, color) in zip(cols, [
            ("🔍 Semantic", sem, "#3b82f6"),
            ("📝 BM25",     bm,  "#f59e0b"),
            ("🔀 Hybrid",   hyb, ACCENT),
        ]):
            with col:
                st.markdown(f"<div style='font-weight:600;color:{color};margin-bottom:8px'>{mode_name}</div>",
                            unsafe_allow_html=True)
                for r in results:
                    icon = CLINICAL_AREA_ICON.get(r.clinical_area, "📄")
                    score_str = f"{r.score:.4f}" if r.score is not None else "—"
                    retriever_tag = f"<span style='color:{ACCENT};font-size:10px'>[{r.retriever}]</span>" \
                        if "hybrid" in r.retriever else ""
                    st.markdown(f"""
                    <div class="result-card">
                      <div style='display:flex;justify-content:space-between;align-items:center'>
                        <span style='font-size:11px;color:#94a3b8'>{icon} {r.doc_type}</span>
                        <span style='font-size:13px;font-weight:700;color:{color}'>{score_str}</span>
                      </div>
                      <div class="result-title">{r.title}</div>
                      <div class="result-snippet">{r.content[:120].replace(chr(10),' ')}…</div>
                      {retriever_tag}
                    </div>
                    """, unsafe_allow_html=True)
    else:
        # ── Single mode ────────────────────────────────────────────
        with st.spinner(f"Searching ({RAG_STRATEGIES[mode]['label']})…"):
            if mode == "semantic":
                results = engine.search_semantic(query, k=k)
            elif mode == "bm25":
                results = engine.search_bm25(query, k=k)
            else:
                results = engine.search_hybrid(query, k=k)

        if not results:
            st.info("No results found for that query.")
            return

        try:
            import altair as alt, pandas as pd
            df = pd.DataFrame({
                "Title": [r.title[:38] for r in results],
                "Score": [r.score or 0.0 for r in results],
            })
            chart = alt.Chart(df).mark_bar(
                cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=ACCENT,
            ).encode(
                x=alt.X("Score:Q", title=""),
                y=alt.Y("Title:N", sort="-x", title=""),
                tooltip=["Title", "Score"],
            ).properties(height=min(220, 40 * len(results)))
            st.altair_chart(chart, use_container_width=True)
        except ImportError:
            pass

        for r in results:
            icon = CLINICAL_AREA_ICON.get(r.clinical_area, "📄")
            score_str = f"{r.score:.4f}" if r.score is not None else "—"
            retriever_html = (
                f"<span style='background:{ACCENT_LIGHT};color:{ACCENT_DARK};"
                f"padding:2px 7px;border-radius:4px;font-size:10px'>{r.retriever}</span>"
                if "hybrid" in r.retriever else ""
            )
            with st.expander(f"{icon}  {r.title}   {score_str}"):
                st.markdown(
                    f"<div style='display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px'>"
                    + chip(r.doc_type) + chip(r.clinical_area)
                    + f"<span class='chip'>vrank={r.vector_rank}</span>"
                    + f"<span class='chip'>brank={r.bm25_rank}</span>"
                    + (retriever_html or "") + "</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(r.content)


# ── Tab 4: Manual Assistant ───────────────────────────────────────────────────

def tab_manual() -> None:
    """
    Render Tab 4 — Manual Assistant.
    RAG chat: retrieves context via the selected strategy, then calls the LLM.
    History persists in session_state; each answer shows an expandable source panel.
    """
    top_left, top_right = st.columns([3, 1])
    with top_left:
        mode_key = st.radio("Strategy", list(RAG_STRATEGIES.keys()),
                            format_func=lambda k: RAG_STRATEGIES[k]["label"],
                            horizontal=True)
    with top_right:
        st.markdown("<br>", unsafe_allow_html=True)
        if not _has_api_key():
            st.markdown(
                f"<div style='background:#fef9c3;border:1px solid #fde68a;border-radius:6px;"
                f"padding:6px 10px;font-size:11px;color:#92400e'>⚠ No API key set</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;"
                f"padding:6px 10px;font-size:11px;color:#166534'>"
                + ("🟢 Nebius" if _nebius_active() else "🟡 OpenAI") + "</div>",
                unsafe_allow_html=True,
            )

    vec_w, bm25_w = 0.6, 0.4
    if mode_key == "hybrid":
        vec_w = st.slider("Vector weight", 0.0, 1.0, 0.6, 0.1, key="manual_vec_w")
        bm25_w = round(1.0 - vec_w, 1)

    if st.button("🗑 Clear chat", key="manual_clear"):
        st.session_state["manual_history"] = []
        st.rerun()

    st.session_state.setdefault("manual_history", [])

    for msg in st.session_state["manual_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                src_lines = " · ".join(
                    f"[{r.retriever}] {r.title[:35]}" for r in msg["sources"][:3]
                )
                st.markdown(
                    f"<div class='chat-sources'>📚 Sources: {src_lines}</div>",
                    unsafe_allow_html=True,
                )

    if prompt := st.chat_input("Describe the clinical issue…"):
        st.session_state["manual_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        engine = get_engine()
        with st.spinner("Retrieving context…"):
            if mode_key == "semantic":
                results = engine.search_semantic(prompt, k=5)
            elif mode_key == "bm25":
                results = engine.search_bm25(prompt, k=5)
            else:
                results = engine.search_hybrid(prompt, k=5, bm25_weight=bm25_w, vector_weight=vec_w)

        context = "\n\n---\n\n".join(f"**{r.title}**\n{r.content[:400]}" for r in results)

        with st.chat_message("assistant"):
            if not _has_api_key():
                answer = "⚠️ Set OPENAI_API_KEY or NEBIUS_API_KEY in .env to get AI answers."
            else:
                from track2_retrieval_engine import build_llm
                from langchain_core.prompts import PromptTemplate
                llm = build_llm()
                chain = PromptTemplate.from_template(RAG_PROMPT) | llm
                with st.spinner("Generating answer…"):
                    response = chain.invoke({"context": context, "question": prompt})
                    answer = response.content if hasattr(response, "content") else str(response)
            st.markdown(answer)

            src_lines = " · ".join(
                f"[{r.retriever}] {r.title[:35]}" for r in results[:3]
            )
            st.markdown(
                f"<div class='chat-sources'>📚 Sources: {src_lines}</div>",
                unsafe_allow_html=True,
            )

        st.session_state["manual_history"].append({
            "role": "assistant", "content": answer, "sources": results,
        })


# ── Tab 5: AI Agent ───────────────────────────────────────────────────────────

def tab_agent() -> None:
    """
    Render Tab 5 — AI Agent.
    Stateful chat backed by ClinicalAgent with a tool-call timeline for each response.
    Shows a PHI warning when the query contains an MRN pattern.
    """
    # Status bar
    backend_color = SUCCESS if _nebius_active() else (WARNING_COL if _has_api_key() else DANGER)
    backend_label = ("🟢 Nebius AI Studio — PHI-safe" if _nebius_active()
                     else ("🟡 OpenAI Cloud — dev only" if _has_api_key()
                           else "🔴 No API key — agent disabled"))
    st.markdown(
        f"<div style='background:#111111;border:1px solid #2d2d2d;border-radius:8px;"
        f"padding:10px 16px;display:flex;justify-content:space-between;align-items:center;"
        f"margin-bottom:12px;font-size:13px'>"
        f"<span><b>ReAct Agent</b> — autonomously selects tools · 5 tools available</span>"
        f"<span style='color:{backend_color};font-weight:600'>{backend_label}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if not _has_api_key():
        st.warning("Set OPENAI_API_KEY or NEBIUS_API_KEY in .env to enable the agent.")

    if st.button("🗑 Clear chat", key="agent_clear"):
        st.session_state["agent_history"] = []
        st.rerun()

    st.session_state.setdefault("agent_history", [])

    for msg in st.session_state["agent_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("steps"):
                _render_tool_timeline(msg["steps"], msg.get("query", ""))

    if prompt := st.chat_input("Ask anything — MRN, bug ID, or clinical question…"):
        st.session_state["agent_history"].append({"role": "user", "content": prompt, "query": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            if MRN_PATTERN.search(prompt):
                st.markdown(
                    f"<div class='phi-banner'>⚠️ <b>PHI detected</b> — MRN identifier present. "
                    f"Ensure Nebius mode is active before using real patient data.</div>",
                    unsafe_allow_html=True,
                )

        with st.chat_message("assistant"):
            if not _has_api_key():
                st.warning("Set API key in .env to run the agent.")
                return

            agent = get_agent()
            if agent is None:
                st.error("Agent failed to initialise. Check terminal logs.")
                return
            with st.spinner("Agent reasoning through tools…"):
                result = agent.run(prompt)

            answer = result.get("output", "")
            steps  = result.get("intermediate_steps", [])
            st.markdown(answer)
            _render_tool_timeline(steps, prompt)

        st.session_state["agent_history"].append({
            "role": "assistant", "content": answer, "steps": steps, "query": prompt,
        })


def _render_tool_timeline(steps: list, query: str) -> None:
    """
    Render a visual tool-call timeline below an agent response.
    Each tool call shows icon, name, input, and a truncated observation.
    Also renders a PHI warning if an MRN is detected in the query.
    """
    if not steps:
        return

    with st.expander(f"🔧 Tool calls — {len(steps)} step{'s' if len(steps)!=1 else ''}", expanded=False):
        for i, (action, obs) in enumerate(steps):
            tool    = action.tool
            inp     = str(action.tool_input)
            obs_str = str(obs)[:200].replace("\n", " ")
            icon    = TOOL_ICON.get(tool, "🔹")
            color   = TOOL_COLOR.get(tool, ACCENT)

            st.markdown(f"""
            <div class="tool-step">
              <div class="tool-icon" style="background:{color}">{icon}</div>
              <div style="flex:1;min-width:0">
                <div class="tool-name" style="color:{color}">
                  {i+1}. {tool}
                </div>
                <div class="tool-input">Input: {inp[:120]}</div>
                <div class="tool-obs">↳ {obs_str}…</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Cited KB sources
        cited = []
        for action, obs in steps:
            if "search_kb" in action.tool:
                for line in str(obs).splitlines():
                    if ". [" in line:
                        cited.append(line.strip())
        if cited:
            st.markdown(
                "<div class='chat-sources'>📚 KB sources referenced:<br>"
                + "<br>".join(dict.fromkeys(cited)[:5])
                + "</div>",
                unsafe_allow_html=True,
            )


# ── Tab 6: PDF Ingestor ───────────────────────────────────────────────────────

def tab_pdf() -> None:
    """
    Render Tab 6 — PDF Ingestor.
    Multi-file uploader with per-file metadata. On submit, chunks and upserts
    each PDF then rebuilds the BM25 index so all tabs see the new content.
    """
    st.markdown(
        f"<div class='phi-banner'>⚠️ <b>PHI Warning:</b> Do not upload documents containing "
        f"real patient data unless Nebius mode is active.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    if not uploaded:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;color:#94a3b8;
          border:2px dashed #2d2d2d;border-radius:12px;background:#0a0a0a'>
          <div style='font-size:48px'>📄</div>
          <div style='font-size:15px;margin-top:8px'>Drop PDF files above to ingest them</div>
          <div style='font-size:12px;margin-top:4px'>
            They'll be chunked, embedded, and added to the live vector store + BM25 index
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown(f"**{len(uploaded)} file{'s' if len(uploaded)>1 else ''} selected**")
    st.markdown("---")

    doc_type_opts = ["product_doc", "runbook", "past_ticket", "faq", "bug_report"]
    area_opts     = list(CLINICAL_AREA_ICON.keys()) + ["general"]

    configs: list[dict] = []
    for f in uploaded:
        with st.expander(f"⚙️ {f.name}  —  configure metadata"):
            c1, c2 = st.columns(2)
            with c1:
                dt   = st.selectbox("Doc type",      doc_type_opts, key=f"dt_{f.name}")
                area = st.selectbox("Clinical area",  area_opts,     key=f"area_{f.name}")
            with c2:
                pri  = st.selectbox("Priority",       ["P3","P2","P1","P0"], key=f"pri_{f.name}")
                tier = st.selectbox("Patient tier",   ["all","regular","plus"], key=f"tier_{f.name}")
        configs.append({"file": f, "doc_type": dt, "clinical_area": area,
                        "priority": pri, "patient_tier": tier})

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬆️ Ingest PDFs", use_container_width=True):
        if not _has_api_key():
            st.error("Set OPENAI_API_KEY or NEBIUS_API_KEY in .env before ingesting.")
            return

        import tempfile
        from track2_pdf_ingestor import PDFIngestor
        engine   = get_engine()
        ingestor = PDFIngestor(engine._vector_store)
        bar      = st.progress(0, text="Starting ingestion…")
        results: list[str] = []

        for i, cfg in enumerate(configs):
            bar.progress((i) / len(configs), text=f"Processing {cfg['file'].name}…")
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(cfg["file"].read())
                tmp_path = Path(tmp.name)
            try:
                n = ingestor.ingest_file(
                    tmp_path,
                    doc_type=cfg["doc_type"], clinical_area=cfg["clinical_area"],
                    priority=cfg["priority"], patient_tier=cfg["patient_tier"],
                )
                results.append(("ok", cfg["file"].name, f"{n} chunks ingested"))
            except Exception as exc:
                results.append(("err", cfg["file"].name, str(exc)))
            finally:
                tmp_path.unlink(missing_ok=True)

        bar.progress(1.0, text="Rebuilding BM25 index…")
        engine.rebuild_bm25()
        bar.empty()

        for status, fname, msg in results:
            if status == "ok":
                st.success(f"✅ **{fname}** — {msg}")
            else:
                st.error(f"❌ **{fname}** — {msg}")

        st.info("✅ BM25 index rebuilt. New documents are searchable in all tabs immediately.")


# ── App entry point ───────────────────────────────────────────────────────────

def main() -> None:
    """
    App entry point — configures page, renders hero header, and routes tabs.
    """
    st.set_page_config(
        page_title="Healthcare-App KB",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed",   # sidebar unused — filters are inline
    )
    inject_css()

    docs = load_docs()
    render_hero(docs)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard",
        "📚 KB Browser",
        "🔬 Search Lab",
        "💬 Assistant",
        "🤖 AI Agent",
        "📄 PDF Ingestor",
    ])

    with tab1: tab_dashboard(docs)
    with tab2: tab_kb(docs)
    with tab3: tab_search()
    with tab4: tab_manual()
    with tab5: tab_agent()
    with tab6: tab_pdf()


if __name__ == "__main__":
    main()
