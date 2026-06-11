"""
track2_retrieval_engine.py

Three-mode retrieval engine for the Healthcare-App Clinical Knowledge Assistant.
Supports semantic (vector), BM25 keyword, and hybrid (RRF) search.
Includes build_llm() factory for Nebius AI Studio / OpenAI cloud switching.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

NAMESPACE = "healthcare-app-rag"
EMBED_MODEL_OPENAI = "text-embedding-3-small"
EMBED_DIMS_OPENAI = 512
EMBED_MODEL_HF = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION = "healthcare-app-track2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RRF_K = 60  # standard constant in Reciprocal Rank Fusion


@dataclass
class RetrievalResult:
    """
    Single retrieval hit returned by all three search modes.

    Attributes:
        title: Document title from the knowledge base.
        doc_type: Category — past_ticket, runbook, product_doc, faq, or bug_report.
        clinical_area: Clinical domain — admissions, medications, lab_results, etc.
        content: Full chunk text as stored in the vector / BM25 index.
        score: Cosine similarity (semantic), BM25 tf-idf value, or RRF fused score.
        bm25_rank: Zero-based rank within BM25 results (None for pure semantic hits).
        vector_rank: Zero-based rank within vector results (None for pure BM25 hits).
        retriever: Origin tag — semantic | bm25 | hybrid-semantic | hybrid-bm25 | hybrid-both.
        source_file: PDF filename when the chunk came from PDF ingestion, else None.
        page_number: 1-indexed page from the source PDF; None for JSON KB documents.
    """
    title: str
    doc_type: str
    clinical_area: str
    content: str
    score: float | None
    bm25_rank: int | None
    vector_rank: int | None
    retriever: str  # "semantic"|"bm25"|"hybrid-semantic"|"hybrid-bm25"|"hybrid-both"
    source_file: str | None
    page_number: int | None


def build_llm() -> Any:
    """
    Return a ChatOpenAI instance pointing at Nebius AI Studio if NEBIUS_API_KEY
    is set, otherwise fall back to the OpenAI cloud endpoint.
    Logs which backend is active at INFO level.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError("pip install langchain-openai") from exc

    nebius_key = os.getenv("NEBIUS_API_KEY")
    if nebius_key:
        model = os.getenv("NEBIUS_MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
        logger.info("LLM backend: Nebius AI Studio (%s)", model)
        return ChatOpenAI(
            model=model,
            openai_api_base="https://api.studio.nebius.com/v1",
            openai_api_key=nebius_key,
            temperature=0,
        )
    model = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-mini")
    logger.info("LLM backend: OpenAI cloud (%s)", model)
    return ChatOpenAI(model=model, temperature=0)


class RetrievalEngine:
    """Semantic, BM25, and hybrid retrieval over the Healthcare-App knowledge base."""

    def __init__(self, kb_path: str | Path, use_pinecone: bool = True) -> None:
        """
        Load the knowledge base, split all documents into overlapping chunks,
        initialise the active vector store (Pinecone when env keys are present,
        ChromaDB otherwise), and build the in-memory BM25 index.

        Args:
            kb_path: Path to healthcare_app_knowledge_base.json.
            use_pinecone: Pass False to force ChromaDB regardless of env vars.
                          Useful for local testing without Pinecone credentials.
        """
        self._kb_path = Path(kb_path)
        self._all_chunks: list[Any] = []
        self._vector_store: Any = None
        self._store_type: str = ""
        self._bm25: Any = None

        raw_docs = self._load_kb_as_docs()
        self._all_chunks = self._chunk_docs(raw_docs)
        self._vector_store, self._store_type = self._init_vector_store(use_pinecone)
        self._bm25 = self._build_bm25(self._all_chunks)
        logger.info(
            "RetrievalEngine ready — store=%s, chunks=%d",
            self._store_type,
            len(self._all_chunks),
        )

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load_kb_as_docs(self) -> list[Any]:
        """Load KB JSON and convert each entry to a LangChain Document."""
        try:
            from langchain_core.documents import Document
        except ImportError as exc:
            raise ImportError("pip install langchain-core") from exc

        records: list[dict] = json.loads(self._kb_path.read_text(encoding="utf-8"))
        docs = []
        for rec in records:
            docs.append(Document(
                page_content=f"{rec['title']}\n\n{rec['content']}",
                metadata={
                    "id": rec["id"],
                    "title": rec["title"],
                    "doc_type": rec["doc_type"],
                    "clinical_area": rec["clinical_area"],
                    "priority": rec["priority"],
                    "platform": rec["platform"],
                    "patient_tier": rec["patient_tier"],
                    "status": rec["status"],
                    "source_file": None,
                    "page_number": None,
                },
            ))
        return docs

    def _chunk_docs(self, docs: list[Any]) -> list[Any]:
        """Split documents into overlapping chunks, preserving metadata."""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            try:
                from langchain.text_splitter import RecursiveCharacterTextSplitter
            except ImportError as exc:
                raise ImportError("pip install langchain-text-splitters") from exc

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{chunk.metadata['id']}_{i}"
        return chunks

    # ── Vector store ─────────────────────────────────────────────────────────

    def _init_vector_store(self, use_pinecone: bool) -> tuple[Any, str]:
        """Try Pinecone; fall back to ChromaDB if keys are absent or init fails."""
        has_keys = bool(os.getenv("PINECONE_API_KEY") and os.getenv("PINECONE_INDEX_NAME"))
        if use_pinecone and has_keys:
            try:
                return self._try_pinecone()
            except Exception as exc:
                logger.warning("Pinecone init failed (%s) — falling back to ChromaDB", exc)
        return self._init_chroma()

    def _try_pinecone(self) -> tuple[Any, str]:
        """Connect to Pinecone; upsert chunks only when the namespace is empty."""
        try:
            from langchain_openai import OpenAIEmbeddings
            from langchain_pinecone import PineconeVectorStore
            from pinecone import Pinecone
        except ImportError as exc:
            raise ImportError("pip install langchain-pinecone pinecone-client langchain-openai") from exc

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
        stats = index.describe_index_stats()
        ns_info = (stats.namespaces or {}).get(NAMESPACE)
        ns_count = ns_info.vector_count if ns_info else 0

        embeddings = OpenAIEmbeddings(model=EMBED_MODEL_OPENAI, dimensions=EMBED_DIMS_OPENAI)
        vs = PineconeVectorStore(index=index, embedding=embeddings, namespace=NAMESPACE)

        if ns_count == 0:
            logger.info("Pinecone namespace empty — upserting %d chunks", len(self._all_chunks))
            vs.add_documents(self._all_chunks)
        else:
            logger.info("Pinecone namespace has %d vectors — skipping upsert", ns_count)
        return vs, "pinecone"

    def _init_chroma(self) -> tuple[Any, str]:
        """Init ChromaDB with sentence-transformers embeddings (384-dim, cosine)."""
        try:
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ImportError(
                "pip install chromadb langchain-chroma langchain-huggingface sentence-transformers"
            ) from exc

        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        logger.info("Pinecone unavailable — falling back to ChromaDB at %s", persist_dir)

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_HF,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vs = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=embeddings,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )
        existing = self._chroma_count(vs)
        if existing == 0:
            logger.info("Chroma collection empty — upserting %d chunks", len(self._all_chunks))
            vs.add_documents(self._all_chunks)
        else:
            logger.info("Chroma collection has %d vectors — skipping upsert", existing)
        return vs, "chroma"

    @staticmethod
    def _chroma_count(vs: Any) -> int:
        """Return the number of vectors already in the ChromaDB collection, or 0 on error."""
        try:
            return vs._collection.count()
        except Exception:
            return 0

    # ── BM25 ─────────────────────────────────────────────────────────────────

    def _build_bm25(self, chunks: list[Any]) -> Any:
        """Build BM25Retriever from document chunks."""
        try:
            from langchain_community.retrievers import BM25Retriever
        except ImportError as exc:
            raise ImportError("pip install langchain-community rank-bm25") from exc

        retriever = BM25Retriever.from_documents(chunks)
        retriever.k = max(20, len(chunks))  # expose all docs for manual top-k slicing
        return retriever

    def rebuild_bm25(self, extra_docs: list[Any] | None = None) -> None:
        """Rebuild BM25 after PDF ingestion. extra_docs are appended before fit."""
        combined = list(self._all_chunks)
        if extra_docs:
            combined.extend(extra_docs)
            self._all_chunks = combined
        self._bm25 = self._build_bm25(combined)
        logger.info("BM25 rebuilt with %d chunks", len(combined))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _doc_to_result(
        self,
        doc: Any,
        score: float | None = None,
        bm25_rank: int | None = None,
        vector_rank: int | None = None,
        retriever: str = "semantic",
    ) -> RetrievalResult:
        """Convert a LangChain Document and its ranking metadata into a RetrievalResult dataclass."""
        m = doc.metadata
        return RetrievalResult(
            title=m.get("title", ""),
            doc_type=m.get("doc_type", ""),
            clinical_area=m.get("clinical_area", ""),
            content=doc.page_content,
            score=score,
            bm25_rank=bm25_rank,
            vector_rank=vector_rank,
            retriever=retriever,
            source_file=m.get("source_file"),
            page_number=m.get("page_number"),
        )

    @staticmethod
    def _content_key(r: "RetrievalResult") -> str:
        """Stable deduplication key across retrievers."""
        return r.title + r.content[:100]

    # ── Public search API ────────────────────────────────────────────────────

    def search_semantic(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """Cosine similarity search via vector store. Returns top-k with score."""
        if not self._vector_store:
            return []
        try:
            hits = self._vector_store.similarity_search_with_score(query, k=k)
        except Exception as exc:
            logger.warning("Semantic search error: %s", exc)
            return []

        results = []
        for rank, (doc, raw_score) in enumerate(hits):
            # Chroma returns cosine distance [0,1]; Pinecone returns similarity [0,1]
            score = (1.0 - float(raw_score)) if self._store_type == "chroma" else float(raw_score)
            results.append(
                self._doc_to_result(doc, score=round(score, 4), vector_rank=rank, retriever="semantic")
            )
        return results

    def search_bm25(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """BM25 keyword search with explicit scores. Returns top-k non-zero results."""
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("pip install numpy") from exc

        scores = self._bm25.vectorizer.get_scores(query.split())
        top_idx = np.argsort(scores)[::-1]

        results: list[RetrievalResult] = []
        rank = 0
        for idx in top_idx:
            if rank >= k:
                break
            sc = float(scores[idx])
            if sc <= 0:
                break
            doc = self._bm25.docs[idx]
            results.append(
                self._doc_to_result(doc, score=round(sc, 4), bm25_rank=rank, retriever="bm25")
            )
            rank += 1
        return results

    def search_hybrid(
        self,
        query: str,
        k: int = 5,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ) -> list[RetrievalResult]:
        """RRF fusion of semantic + BM25. Returns top-k with fused score and source tags."""
        sem = self.search_semantic(query, k=k * 3)
        bm25 = self.search_bm25(query, k=k * 3)
        return self._rrf_fuse(sem, bm25, k, bm25_weight, vector_weight)

    def _rrf_fuse(
        self,
        sem: list[RetrievalResult],
        bm25: list[RetrievalResult],
        k: int,
        bm25_w: float,
        vec_w: float,
    ) -> list[RetrievalResult]:
        """Reciprocal Rank Fusion: score_i = vec_w/(RRF_K+vrank) + bm25_w/(RRF_K+brank)."""
        rrf: dict[str, float] = {}
        sem_map: dict[str, RetrievalResult] = {}
        bm25_map: dict[str, RetrievalResult] = {}

        for rank, r in enumerate(sem):
            key = self._content_key(r)
            rrf[key] = rrf.get(key, 0.0) + vec_w / (RRF_K + rank)
            sem_map[key] = r

        for rank, r in enumerate(bm25):
            key = self._content_key(r)
            rrf[key] = rrf.get(key, 0.0) + bm25_w / (RRF_K + rank)
            bm25_map[key] = r

        sorted_keys = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:k]
        fused: list[RetrievalResult] = []
        for key in sorted_keys:
            in_sem, in_bm25 = key in sem_map, key in bm25_map
            base = sem_map.get(key) or bm25_map[key]
            tag = "hybrid-both" if (in_sem and in_bm25) else ("hybrid-semantic" if in_sem else "hybrid-bm25")
            v_rank = next((i for i, r in enumerate(sem) if self._content_key(r) == key), None)
            b_rank = next((i for i, r in enumerate(bm25) if self._content_key(r) == key), None)
            fused.append(RetrievalResult(
                title=base.title, doc_type=base.doc_type, clinical_area=base.clinical_area,
                content=base.content, score=round(rrf[key], 6),
                bm25_rank=b_rank, vector_rank=v_rank, retriever=tag,
                source_file=base.source_file, page_number=base.page_number,
            ))
        return fused


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    KB = Path(__file__).parent / "healthcare_app_knowledge_base.json"
    engine = RetrievalEngine(KB)

    query = "stuck lab results for patient MRN-293847 — HL7 routing error"
    print(f"\n{'='*64}\nQuery: {query}\n{'='*64}")

    print("\n🔍 SEMANTIC (top-5):")
    for r in engine.search_semantic(query, k=5):
        print(f"  [{r.score:.4f}] {r.title}  ({r.doc_type}/{r.clinical_area})")

    print("\n📝 BM25 (top-5):")
    for r in engine.search_bm25(query, k=5):
        print(f"  [{r.score:.4f}] {r.title}  ({r.doc_type}/{r.clinical_area})")

    print("\n🔀 HYBRID / RRF (top-5):")
    for r in engine.search_hybrid(query, k=5):
        print(f"  [{r.score:.6f}] [{r.retriever}] {r.title}")
        print(f"    vrank={r.vector_rank}  brank={r.bm25_rank}")
