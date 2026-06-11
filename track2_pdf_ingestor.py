"""
track2_pdf_ingestor.py

PDF ingestion module for the Healthcare-App Clinical Knowledge Assistant.
Extracts text from PDFs using PyMuPDF (primary) or pdfplumber (fallback),
chunks, and upserts into the active vector store (Pinecone or ChromaDB).
BM25 is rebuilt after each ingestion via engine.rebuild_bm25().
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HEADER_FOOTER_RE = re.compile(
    r"^(Page\s+\d+|Healthcare-App\s+Confidential|DRAFT)\s*$",
    re.IGNORECASE,
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


class PDFIngestor:
    """Extract, chunk, embed, and upsert PDF content into the active vector store."""

    def __init__(
        self,
        vector_store: Any,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        """
        Args:
            vector_store: Active Pinecone or ChromaDB vector store instance.
                          Pass engine._vector_store from an initialised RetrievalEngine.
            chunk_size: Maximum characters per text chunk. Defaults to 500.
            chunk_overlap: Character overlap between consecutive chunks to preserve
                           context across chunk boundaries. Defaults to 100.
        """
        self._vs = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._splitter = self._build_splitter()

    def _build_splitter(self) -> Any:
        """Instantiate RecursiveCharacterTextSplitter with the configured chunk size and overlap."""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap
        )

    # ── PDF loading ───────────────────────────────────────────────────────────

    def _load_with_pymupdf(self, pdf_path: Path) -> list[Any]:
        """Primary loader: PyMuPDF (fitz)."""
        try:
            from langchain_community.document_loaders import PyMuPDFLoader
        except ImportError as exc:
            raise ImportError("pip install pymupdf langchain-community") from exc
        return PyMuPDFLoader(str(pdf_path)).load()

    def _load_with_pdfplumber(self, pdf_path: Path) -> list[Any]:
        """Fallback loader: pdfplumber."""
        try:
            from langchain_community.document_loaders import PDFPlumberLoader
        except ImportError as exc:
            raise ImportError("pip install pdfplumber langchain-community") from exc
        return PyMuPDFLoader(str(pdf_path)).load()  # noqa: F821 — reached only if PyMuPDF fails

    def _load_pdf(self, pdf_path: Path) -> list[Any]:
        """Try PyMuPDF first; fall back to pdfplumber."""
        try:
            return self._load_with_pymupdf(pdf_path)
        except ImportError:
            logger.info("PyMuPDF not available — using pdfplumber for %s", pdf_path.name)
            try:
                from langchain_community.document_loaders import PDFPlumberLoader
                return PDFPlumberLoader(str(pdf_path)).load()
            except ImportError as exc:
                raise ImportError("pip install pymupdf  # or: pip install pdfplumber") from exc

    # ── Text cleaning ─────────────────────────────────────────────────────────

    @staticmethod
    def _clean_page(text: str) -> str:
        """Strip header/footer lines and normalise whitespace."""
        lines = text.splitlines()
        cleaned = [
            line for line in lines
            if not _HEADER_FOOTER_RE.match(line.strip())
        ]
        return "\n".join(cleaned).strip()

    # ── Core ingestion ────────────────────────────────────────────────────────

    def ingest_file(
        self,
        pdf_path: str | Path,
        doc_type: str = "product_doc",
        clinical_area: str = "general",
        priority: str = "P3",
        patient_tier: str = "all",
        status: str = "active",
    ) -> int:
        """
        Extract text from pdf_path, chunk, embed, and upsert into the vector store.
        Returns the number of chunks upserted.
        Metadata: title, doc_type, clinical_area, priority, platform, patient_tier,
                  status, source_file, page_number.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages = self._load_pdf(pdf_path)
        logger.info("Loaded %d pages from %s", len(pages), pdf_path.name)

        base_meta = {
            "title": pdf_path.stem.replace("_", " ").replace("-", " ").title(),
            "doc_type": doc_type,
            "clinical_area": clinical_area,
            "priority": priority,
            "platform": "all",
            "patient_tier": patient_tier,
            "status": status,
            "source_file": pdf_path.name,
        }

        docs_to_chunk = self._pages_to_docs(pages, base_meta, pdf_path.name)
        chunks = self._splitter.split_documents(docs_to_chunk)

        if not chunks:
            logger.warning("No content extracted from %s", pdf_path.name)
            return 0

        self._vs.add_documents(chunks)
        logger.info("Upserted %d chunks from %s", len(chunks), pdf_path.name)
        return len(chunks)

    def _pages_to_docs(
        self, pages: list[Any], base_meta: dict, filename: str
    ) -> list[Any]:
        """Convert loaded pages to Documents with cleaned text and page metadata."""
        try:
            from langchain_core.documents import Document
        except ImportError as exc:
            raise ImportError("pip install langchain-core") from exc

        docs = []
        for page in pages:
            cleaned = self._clean_page(page.page_content)
            if not cleaned:
                continue
            page_num = page.metadata.get("page", page.metadata.get("page_number", None))
            if isinstance(page_num, int):
                page_num += 1  # make 1-indexed if 0-indexed
            meta = {**base_meta, "page_number": page_num, "id": f"{filename}_p{page_num}"}
            docs.append(Document(page_content=cleaned, metadata=meta))
        return docs

    def ingest_folder(
        self,
        folder_path: str | Path,
        clinical_area_map: dict[str, str] | None = None,
    ) -> dict[str, int]:
        """
        Ingest all PDFs in a folder.
        clinical_area_map: {filename_stem: clinical_area} for custom mapping.
        Returns {filename: chunks_upserted}.
        """
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder_path}")

        pdfs = sorted(folder_path.glob("*.pdf"))
        if not pdfs:
            logger.info("No PDFs found in %s", folder_path)
            return {}

        results: dict[str, int] = {}
        for pdf in pdfs:
            area = (clinical_area_map or {}).get(pdf.stem, "general")
            try:
                n = self.ingest_file(pdf, clinical_area=area)
                results[pdf.name] = n
            except Exception as exc:
                logger.error("Failed to ingest %s: %s", pdf.name, exc)
                results[pdf.name] = 0
        return results


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sample_dir = Path(__file__).parent / "sample_pdfs"

    if not sample_dir.exists() or not list(sample_dir.glob("*.pdf")):
        print("Usage example:")
        print("  from track2_retrieval_engine import RetrievalEngine")
        print("  from track2_pdf_ingestor import PDFIngestor")
        print()
        print("  engine = RetrievalEngine('healthcare_app_knowledge_base.json')")
        print("  ingestor = PDFIngestor(engine._vector_store)")
        print("  n = ingestor.ingest_file('my_policy.pdf', doc_type='product_doc',")
        print("                           clinical_area='admissions', priority='P2')")
        print("  engine.rebuild_bm25()  # update BM25 with newly ingested chunks")
        print("  print(f'Ingested {n} chunks')")
        print()
        print(f"Place PDF files in {sample_dir} and re-run to demo live ingestion.")
        sys.exit(0)

    # Live demo if PDFs are present
    sys.path.insert(0, str(Path(__file__).parent))
    from track2_retrieval_engine import RetrievalEngine

    KB = Path(__file__).parent / "healthcare_app_knowledge_base.json"
    engine = RetrievalEngine(KB)
    ingestor = PDFIngestor(engine._vector_store)

    results = ingestor.ingest_folder(sample_dir)
    engine.rebuild_bm25()

    print("\nIngestion results:")
    for fname, count in results.items():
        print(f"  {fname}: {count} chunks")
