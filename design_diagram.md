# Healthcare-App — System Design Diagrams

## 1. Component Architecture

```mermaid
graph TB
    subgraph UI["Streamlit App — track2_kb_viewer.py"]
        T1[Tab 1\nKnowledge Base Browser]
        T2[Tab 2\nSearch Lab]
        T3[Tab 3\nManual Assistant]
        T4[Tab 4\nAI Agent]
        T5[Tab 5\nPDF Ingestor]
    end

    subgraph Agent["ReAct Agent — track2_clinical_agent.py"]
        AE[AgentExecutor\nmax_iterations=6]
        S1[Tool: search_kb_semantic]
        S2[Tool: search_kb_bm25]
        S3[Tool: search_kb_hybrid]
        S4[Tool: lookup_patient_record\nmock Epic FHIR API]
        S5[Tool: check_system_status\nmock incident API]
        AE --> S1 & S2 & S3 & S4 & S5
    end

    subgraph Retrieval["Retrieval Engine — track2_retrieval_engine.py"]
        RE[RetrievalEngine]
        SEM[search_semantic\ncosine similarity]
        BM[search_bm25\nIDF keyword scoring]
        HYB[search_hybrid\nRRF fusion 0.6/0.4]
        LLM[build_llm\nNebius or OpenAI]
        RE --> SEM & BM & HYB
    end

    subgraph VectorStore["Vector Store"]
        PC[(Pinecone\n512-dim cosine\ntext-embedding-3-small)]
        CH[(ChromaDB\n384-dim cosine\nall-MiniLM-L6-v2)]
        PC -. "fallback if no key" .-> CH
    end

    subgraph KB["Knowledge Base"]
        JSON[healthcare_app_knowledge_base.json\n50 clinical documents\npast_ticket×18 runbook×10\nproduct_doc×10 faq×7 bug_report×5]
    end

    subgraph PDF["PDF Ingestor — track2_pdf_ingestor.py"]
        PI[PDFIngestor\nPyMuPDF → pdfplumber]
        CL[Header/Footer Stripper\nRecursiveCharacterTextSplitter\nchunk_size=500 overlap=100]
        PI --> CL
    end

    T4 --> AE
    T2 & T3 --> RE
    T5 --> PI
    S1 & S2 & S3 --> RE
    RE --> VectorStore
    RE --> BM
    JSON --> RE
    CL --> VectorStore
    CL --> RE
    LLM --> AE
```

---

## 2. Data Flow — Query Processing

```mermaid
sequenceDiagram
    actor CC as Care Coordinator
    participant UI as Streamlit App
    participant AG as ClinicalAgent
    participant RE as RetrievalEngine
    participant VS as Vector Store<br/>(Pinecone/Chroma)
    participant BM as BM25 Index
    participant LLM as LLM<br/>(Nebius/OpenAI)

    CC->>UI: Enter query (e.g. "MRN-334521 duplicate metoprolol")
    UI->>UI: Detect MRN pattern → show PHI warning
    UI->>AG: agent.run(query)

    AG->>LLM: Thought: which tool to call?
    LLM-->>AG: Action: lookup_patient_record(MRN-334521)
    AG->>AG: Observation: MAR-DISCREPANCY flag, two metoprolol orders

    LLM-->>AG: Action: search_kb_bm25("MRN-334521 metoprolol BUG-EHR-2301")
    AG->>RE: search_bm25(query, k=5)
    RE->>BM: get_scores(tokens)
    BM-->>RE: ranked doc indices + scores
    RE-->>AG: [RetrievalResult ×5]

    LLM-->>AG: Action: check_system_status(BUG-EHR-2301)
    AG->>AG: Observation: status=active, workaround=manual override

    LLM-->>AG: Action: search_kb_hybrid("medication reconciliation")
    AG->>RE: search_hybrid(query, k=5)
    RE->>VS: similarity_search_with_score(query, k=15)
    VS-->>RE: [(doc, cosine_score) ×15]
    RE->>BM: get_scores(tokens)
    BM-->>RE: bm25 scores
    RE->>RE: RRF fusion: score = 0.6/(60+vrank) + 0.4/(60+brank)
    RE-->>AG: [RetrievalResult ×5 with hybrid-both/hybrid-semantic/hybrid-bm25 tags]

    LLM-->>AG: Final Answer (cited, multi-source)
    AG-->>UI: {output, intermediate_steps}
    UI-->>CC: Streaming answer + tool-call expanders
```

---

## 3. Retrieval Mode Comparison

```mermaid
graph LR
    Q[User Query] --> SEM & BM25 & HYB

    subgraph SEM["Semantic Search"]
        direction TB
        E1[Embed query\ntext-embedding-3-small\nor all-MiniLM-L6-v2]
        E2[Cosine similarity\nagainst all KB vectors]
        E3[Return top-k\nby similarity score]
        E1 --> E2 --> E3
    end

    subgraph BM25["BM25 Keyword Search"]
        direction TB
        B1[Tokenize query\nword splitting]
        B2[IDF weighting\nrare tokens score higher]
        B3[Return top-k\nby BM25 score]
        B1 --> B2 --> B3
    end

    subgraph HYB["Hybrid RRF Fusion"]
        direction TB
        H1[Run Semantic\ntop-3k candidates]
        H2[Run BM25\ntop-3k candidates]
        H3["RRF score =\nvec_w/(60+vrank)\n+ bm25_w/(60+brank)"]
        H4[Deduplicate\nlabel: hybrid-both\nhybrid-semantic\nhybrid-bm25]
        H1 & H2 --> H3 --> H4
    end

    E3 -->|"Best for: synonyms\nparaphrase, intent"| OUT
    B3 -->|"Best for: MRN, ERR_*\nBUG-*, exact codes"| OUT
    H4 -->|"Best for: mixed queries\nwith both"| OUT
    OUT[Top-k RetrievalResults]
```

---

## 4. LLM Backend Selection

```mermaid
flowchart TD
    START([build_llm called]) --> CHECK{NEBIUS_API_KEY\nset in .env?}
    CHECK -- Yes --> NEB["ChatOpenAI\napi_base: https://api.studio.nebius.com/v1\nmodel: NEBIUS_MODEL_NAME\n(default: meta-llama/Meta-Llama-3.1-70B-Instruct)\nPHI-safe ✅"]
    CHECK -- No --> OAI["ChatOpenAI\napi_base: api.openai.com\nmodel: OPENAI_MODEL_NAME\n(default: gpt-4.1-mini)\nDev only — no PHI ⚠️"]
    NEB --> LLM_OUT([LLM instance\ntemperature=0])
    OAI --> LLM_OUT
```

---

## 5. Vector Store Selection

```mermaid
flowchart TD
    START([RetrievalEngine init]) --> CHECK{PINECONE_API_KEY\n+ PINECONE_INDEX_NAME\nboth set?}
    CHECK -- Yes --> TRY[Try Pinecone connection]
    TRY --> OK{Success?}
    OK -- Yes --> PC["Pinecone\ndims=512, metric=cosine\nembedding: text-embedding-3-small\ncloud: aws / us-east-1"]
    OK -- No --> FALL[Log warning\nFall back to ChromaDB]
    CHECK -- No --> FALL
    FALL --> CH["ChromaDB\ndims=384, metric=cosine\nembedding: all-MiniLM-L6-v2\npersist_dir: CHROMA_PERSIST_DIR"]
    PC --> UPSERT{Namespace\nalready has vectors?}
    CH --> UPSERT2{Collection\nalready has vectors?}
    UPSERT -- No --> ADD[Upsert all KB chunks]
    UPSERT -- Yes --> SKIP[Skip upsert\nlog vector count]
    UPSERT2 -- No --> ADD2[Upsert all KB chunks]
    UPSERT2 -- Yes --> SKIP2[Skip upsert\nlog vector count]
```

---

## 6. PDF Ingestion Flow

```mermaid
flowchart TD
    UP[User uploads PDF\nvia Streamlit or CLI] --> PY[PyMuPDF\nextract text per page]
    PY --> FB{Extraction\nsucceeded?}
    FB -- No --> PL[pdfplumber fallback\nextract text per page]
    FB -- Yes --> STRIP
    PL --> STRIP[Strip headers/footers\nregex: Page N, DRAFT,\nHealthcare-App Confidential]
    STRIP --> CHUNK[RecursiveCharacterTextSplitter\nchunk_size=500\nchunk_overlap=100]
    CHUNK --> META[Attach metadata\ndoc_type, clinical_area,\npriority, patient_tier,\nsource_file, page_number]
    META --> VS[(Upsert to\nactive vector store)]
    VS --> BM25[Rebuild BM25 index\nengine.rebuild_bm25]
    BM25 --> DONE[New content\nsearchable via all 3 modes]
```

---

## Environment Variables Quick Reference

| Variable | Default | Effect |
|----------|---------|--------|
| `OPENAI_API_KEY` | — | Enables OpenAI cloud LLM |
| `OPENAI_MODEL_NAME` | `gpt-4.1-mini` | OpenAI model to use |
| `NEBIUS_API_KEY` | — | Activates Nebius AI Studio (PHI-safe) |
| `NEBIUS_MODEL_NAME` | `meta-llama/Meta-Llama-3.1-70B-Instruct` | Nebius model |
| `PINECONE_API_KEY` | — | Enables Pinecone cloud vector store |
| `PINECONE_INDEX_NAME` | — | Pinecone index (dims=512, cosine) |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistence directory |
