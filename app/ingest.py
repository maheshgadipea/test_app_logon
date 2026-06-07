"""
Load the reference .docx, chunk it, embed with Gemini text-embedding-004,
and persist a Chroma vector store on disk.

This runs ONCE at app startup (see main.py lifespan). The vector store
object is then held in module state and reused for every /chat request —
we never re-embed per request, and we never rebuild from scratch unless
the persisted Chroma directory is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
    REFERENCE_DOC_PATH,
    VECTOR_STORE_DIR,
)

log = logging.getLogger(__name__)

_vector_store: Chroma | None = None


def build_or_load_vector_store() -> Chroma:
    """Build the vector store at startup (once) and keep it in memory.

    Re-uses the persisted Chroma directory across container restarts, so we
    only embed the document the first time. Delete ./storage/chroma to
    force a rebuild (e.g. after changing the doc or chunking settings).
    """
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set. "
            "Add it to your .env or pass via docker-compose."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    # A half-failed previous ingest can leave a NON-EMPTY directory with
    # ZERO vectors. Distinguishing "ready" from "stale" via mere
    # `any(iterdir())` masks that failure. Build the store first, ask
    # Chroma how many vectors it actually has, and only skip re-ingest
    # when there's real data inside.
    store = Chroma(
        collection_name="reference_doc",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_STORE_DIR),
    )
    existing_count = store._collection.count()

    if existing_count == 0:
        if not REFERENCE_DOC_PATH.exists():
            raise FileNotFoundError(
                f"Reference document not found at {REFERENCE_DOC_PATH}. "
                "Drop your .docx / .txt / .md in ./data/ "
                "(or set REFERENCE_DOC_PATH)."
            )
        log.info("Ingesting %s (Chroma was empty)", REFERENCE_DOC_PATH)
        docs = _load_document(REFERENCE_DOC_PATH)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(docs)
        log.info("Indexing %d chunks into Chroma", len(chunks))
        store.add_documents(chunks)
    else:
        log.info(
            "Reusing persisted vector store at %s (%d vectors)",
            VECTOR_STORE_DIR, existing_count,
        )

    _vector_store = store
    return store


def get_vector_store() -> Chroma:
    if _vector_store is None:
        raise RuntimeError(
            "Vector store not initialized — call build_or_load_vector_store() first."
        )
    return _vector_store


# ─── Loader dispatch ────────────────────────────────────────────────────
# Pick the right LangChain loader based on file extension. Add new formats
# here (PDF, HTML, etc.) without touching the rest of the pipeline.
def _load_document(path: Path):
    ext = path.suffix.lower()
    if ext == ".docx":
        return Docx2txtLoader(str(path)).load()
    if ext in (".txt", ".md"):
        return TextLoader(str(path), encoding="utf-8").load()
    raise ValueError(
        f"Unsupported document extension {ext!r} for {path}. "
        "Supported: .docx, .txt, .md (extend _load_document to add more)."
    )
