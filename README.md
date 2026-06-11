# Healthcare-App Clinical Knowledge Assistant
### Track 2 — Hybrid RAG with BM25 + Semantic + ReAct Agent

A clinical knowledge assistant for hospital care coordinators. Combines hybrid retrieval (BM25 + semantic vector search via Reciprocal Rank Fusion) with a ReAct AI agent that can look up patient records, check system status, and answer questions from a 50-document clinical knowledge base.

---

## Architecture

```
Care Coordinator Query
        │
        ▼
┌───────────────────────────────────────────────────────┐
│         ReAct Agent  (build_llm() → Nebius/OpenAI)   │
│  Tool 1: search_kb_semantic  — cosine vector search   │
│  Tool 2: search_kb_bm25      — BM25 keyword scoring   │
│  Tool 3: search_kb_hybrid    — RRF fusion (0.6/0.4)   │
│  Tool 4: lookup_patient_record — mock Epic FHIR API   │
│  Tool 5: check_system_status   — mock incident API    │
└───────────────────────────────────────────────────────┘
        │
  ┌─────┴──────────┐
  ▼                ▼
Pinecone (primary) ChromaDB (fallback)
512-dim cosine     384-dim cosine
text-embedding-3-small  all-MiniLM-L6-v2
  ▲
PDF Ingestor (PyMuPDF → pdfplumber fallback)
```

> See [design_diagram.md](design_diagram.md) for the full Mermaid component + data-flow diagram.

---

## Project Structure

```
healthcare-app/
├── healthcare_app_knowledge_base.json  # 50 clinical KB documents
├── track2_retrieval_engine.py          # Core: 3-mode retrieval + build_llm()
├── track2_pdf_ingestor.py              # PDF ingestion with PyMuPDF
├── track2_clinical_agent.py            # ReAct AgentExecutor with 5 tools
├── track2_kb_viewer.py                 # Streamlit app (5 tabs)
├── track2_hybrid_rag.ipynb             # Teaching notebook (Sections 0–7)
├── requirements.txt
├── .env.example
└── sample_pdfs/                        # Drop PDFs here for ingestion demo
```

---

## Prerequisites

- Python 3.10 or 3.11
- An API key from **at least one** of:
  - [OpenAI](https://platform.openai.com) — for development / non-PHI queries
  - [Nebius AI Studio](https://studio.nebius.com) — for production / PHI-containing queries
- (Optional) [Pinecone](https://app.pinecone.io) account — for cloud vector store. Without it the app falls back to local ChromaDB automatically.

---

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Dev only | OpenAI API key for development queries |
| `OPENAI_MODEL_NAME` | No | Defaults to `gpt-4.1-mini` |
| `NEBIUS_API_KEY` | Production | Activates Nebius AI Studio (PHI-safe) |
| `NEBIUS_MODEL_NAME` | No | Defaults to `meta-llama/Meta-Llama-3.1-70B-Instruct` |
| `PINECONE_API_KEY` | No | Enables Pinecone cloud vector store |
| `PINECONE_INDEX_NAME` | No | Must be created with dims=512, metric=cosine |
| `CHROMA_PERSIST_DIR` | No | Defaults to `./chroma_db` |

**LLM selection logic:** `build_llm()` checks `NEBIUS_API_KEY` first. If set → Nebius AI Studio. Otherwise → OpenAI cloud. No code change needed.

**Vector store selection logic:** If `PINECONE_API_KEY` + `PINECONE_INDEX_NAME` are both set → Pinecone. Otherwise → ChromaDB (local, no key required).

---

## Running the Project

### Smoke test — Retrieval Engine

Verifies all three retrieval modes (semantic, BM25, hybrid) work end-to-end:

```bash
python track2_retrieval_engine.py
```

Expected output: 5 results per mode for the query `"stuck lab results for patient MRN-293847 — HL7 routing error"`.

### Smoke test — PDF Ingestor

```bash
python track2_pdf_ingestor.py
```

With an empty `sample_pdfs/` folder this prints a usage example. To test real ingestion, drop any PDF into `sample_pdfs/` and re-run.

### Smoke test — Clinical Agent

Runs 3 qualifying scenarios through the ReAct agent (requires an API key in `.env`):

```bash
python track2_clinical_agent.py
```

Expected: each case prints intermediate tool calls and a final answer.

### Streamlit App (main UI)

```bash
streamlit run track2_kb_viewer.py
```

Opens at [http://localhost:8501](http://localhost:8501). Five tabs:

| Tab | What it does |
|-----|--------------|
| **Clinical Knowledge Base** | Browse and filter all 50 KB documents |
| **Search Lab** | Compare semantic / BM25 / hybrid results side-by-side |
| **Manual Assistant** | Chat interface with retrieval-backed answers |
| **AI Agent** | Full ReAct agent with streaming output and tool-call expanders |
| **PDF Ingestor** | Upload PDFs, set metadata, ingest into the live vector store |

### Teaching Notebook

```bash
jupyter notebook track2_hybrid_rag.ipynb
# or
jupyter lab track2_hybrid_rag.ipynb
```

Run cells top-to-bottom. Sections 0–6 work without an LLM key (retrieval only). Section 7 (ReAct agent) requires `OPENAI_API_KEY` or `NEBIUS_API_KEY` in `.env`.

To execute all cells non-interactively:

```bash
jupyter nbconvert --to notebook --execute track2_hybrid_rag.ipynb --output track2_hybrid_rag_executed.ipynb
```

---

## Pinecone Setup (optional)

If you want cloud vector storage instead of local ChromaDB:

1. Create a free Pinecone index at [app.pinecone.io](https://app.pinecone.io)
   - **Dimensions:** 512
   - **Metric:** cosine
   - **Cloud / Region:** aws / us-east-1 (or any)
2. Add to `.env`:
   ```
   PINECONE_API_KEY=your-key
   PINECONE_INDEX_NAME=healthcare-app-rag
   ```
3. The engine upserts all KB chunks on first run and skips re-indexing on subsequent runs.

---

## PHI Safety

Queries containing real MRN identifiers (pattern `MRN-\d+`) must be routed through **Nebius AI Studio**, not the OpenAI cloud endpoint, to avoid transmitting PHI to a third-party server.

- Set `NEBIUS_API_KEY` in `.env` to activate the PHI-safe backend.
- The Streamlit AI Agent tab shows a red PHI warning when an MRN is detected in the query.
- `build_llm()` in `track2_retrieval_engine.py` handles the switch automatically — no code changes required.

---

## Key Identifiers in the Knowledge Base

The KB includes realistic clinical identifiers for testing:

| Identifier | Type | Description |
|------------|------|-------------|
| `MRN-334521` | Patient | Duplicate metoprolol 25mg orders — MAR discrepancy |
| `MRN-293847` | Patient | Pending CBC, HL7 routing error |
| `MRN-847392` | Patient | LTC sync issue, AUTH-4012 flag |
| `BUG-EHR-2301` | Bug | Active EHR medication reconciliation bug |
| `BUG-EHR-2201` | Bug | Active EHR lab result display bug |
| `BUG-SCH-0445` | Bug | Active scheduling conflict bug |
| `ERR_HL7_ROUTE_FAIL` | Error code | HL7 message routing failure |
| `AUTH-4012` | Error code | Prior auth verification timeout |
| `SCH-CONFLICT-88` | Error code | Booking conflict error |
| `TC-CONN-FAIL` | Error code | Telehealth connection failure |

---

## Dependency Reference

| Package | Purpose |
|---------|---------|
| `langchain`, `langchain-openai` | LLM chains and ChatOpenAI |
| `langchain-pinecone` | Pinecone vector store integration |
| `langchain-chroma` | ChromaDB vector store integration |
| `langchain-community` | BM25Retriever |
| `langchain-huggingface` | HuggingFace sentence-transformers embeddings |
| `pinecone-client` | Pinecone Python SDK |
| `chromadb` | Local vector database |
| `sentence-transformers` | all-MiniLM-L6-v2 embeddings for ChromaDB |
| `rank-bm25` | BM25 scoring algorithm |
| `pymupdf` | PDF text extraction (primary) |
| `pdfplumber` | PDF text extraction (fallback, better for tables) |
| `streamlit` | Web app UI |
| `altair` | Interactive score charts in Search Lab tab |
| `fpdf2` | Synthetic PDF generation in notebook demo |
| `python-dotenv` | `.env` file loading |
