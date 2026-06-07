"""
Load the reference document, chunk it, embed via the llm-server, and
persist into pgvector inside the postgres container.

Runs ONCE at app startup (see main.py lifespan). The PGVector store is
held in module state and reused for every /chat request — we never
re-embed per request, and we never re-ingest unless the collection is
empty.
"""

from __future__ import annotations

import logging
from pathlib import Path

import psycopg
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    LLM_SERVER_URL,
    POSTGRES_URL,
    REFERENCE_DOC_PATH,
    VECTOR_COLLECTION,
)
from .llm_client import RemoteEmbeddings

log = logging.getLogger(__name__)

_vector_store: PGVector | None = None


def _sqlalchemy_url(uri: str) -> str:
    """PGVector goes through SQLAlchemy; force the psycopg3 driver so it
    doesn't try to pull in psycopg2."""
    if uri.startswith("postgresql+"):
        return uri
    if uri.startswith("postgresql://"):
        return "postgresql+psycopg://" + uri[len("postgresql://") :]
    if uri.startswith("postgres://"):
        return "postgresql+psycopg://" + uri[len("postgres://") :]
    return uri


def _existing_vector_count(collection: str) -> int:
    """Count rows already embedded for this collection. Returns 0 if the
    pgvector tables don't exist yet (first boot)."""
    try:
        with psycopg.connect(POSTGRES_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                      SELECT 1 FROM information_schema.tables
                      WHERE table_name = 'langchain_pg_embedding'
                    )
                    """
                )
                if not cur.fetchone()[0]:
                    return 0
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = %s
                    """,
                    (collection,),
                )
                return int(cur.fetchone()[0] or 0)
    except psycopg.Error as e:
        log.warning("Could not count existing vectors (%s); will re-ingest.", e)
        return 0


def build_or_load_vector_store() -> PGVector:
    """Build the vector store at startup (once) and keep it in memory.

    Re-uses persisted pgvector rows across container restarts, so we only
    embed the document the first time. Drop the rows for our collection
    (or just `DROP TABLE langchain_pg_embedding`) to force a rebuild.
    """
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    embeddings = RemoteEmbeddings(
        base_url=LLM_SERVER_URL,
        model=EMBEDDING_MODEL,
    )

    # Constructing PGVector creates the extension + schema tables if
    # they're missing (idempotent), so this is safe to call every boot.
    store = PGVector(
        embeddings=embeddings,
        collection_name=VECTOR_COLLECTION,
        connection=_sqlalchemy_url(POSTGRES_URL),
        use_jsonb=True,
    )

    existing_count = _existing_vector_count(VECTOR_COLLECTION)

    if existing_count == 0:
        if not REFERENCE_DOC_PATH.exists():
            raise FileNotFoundError(
                f"Reference document not found at {REFERENCE_DOC_PATH}. "
                "Drop your .docx / .txt / .md in ./data/ "
                "(or set REFERENCE_DOC_PATH)."
            )
        log.info("Ingesting %s (pgvector was empty)", REFERENCE_DOC_PATH)
        docs = _load_document(REFERENCE_DOC_PATH)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(docs)
        log.info("Indexing %d chunks into pgvector", len(chunks))
        store.add_documents(chunks)
    else:
        log.info(
            "Reusing persisted pgvector collection %r (%d vectors)",
            VECTOR_COLLECTION, existing_count,
        )

    _vector_store = store
    return store


def get_vector_store() -> PGVector:
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
