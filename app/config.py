"""
Central knobs for the chatbot.

═══════════════════════════════════════════════════════════════════════════
  >>> PLUG IN YOUR REFERENCE DOCUMENT HERE <<<
═══════════════════════════════════════════════════════════════════════════
  1. Drop your .docx into ./data/
  2. Either rename it to `reference.docx` OR override REFERENCE_DOC_PATH
     via the env var (docker-compose.yml already wires the data/ mount).
  3. Tweak CHUNK_SIZE / CHUNK_OVERLAP / TOP_K below if retrieval feels off.
  4. To force a re-ingest, drop the pgvector tables (or just truncate the
     `langchain_pg_embedding` rows for our collection).
═══════════════════════════════════════════════════════════════════════════

The chatbot owns no Google credentials and no on-disk persistence —
all model calls go through the `llm-server` container, and all state
(vector embeddings + LangGraph checkpoints) lives in the `postgres`
container.
"""

from __future__ import annotations

import os
from pathlib import Path


# ─── Document & retrieval settings (tweak freely) ───────────────────────
REFERENCE_DOC_PATH = Path(
    os.getenv("REFERENCE_DOC_PATH", "/app/data/bcb_login.txt")
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "4"))


# ─── Remote LLM server ──────────────────────────────────────────────────
# The chatbot calls this URL for ALL model interactions — both chat
# generation (gemini-2.5-flash) and embeddings (gemini-embedding-001).
LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://llm-server:8100")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.4"))


# ─── Postgres (pgvector + LangGraph checkpoints) ────────────────────────
# Used by BOTH the vector store (langchain_postgres.PGVector) and the
# checkpointer (langgraph.checkpoint.postgres.PostgresSaver). One database,
# different tables.
POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://chatbot:chatbot@postgres:5432/chatbot",
)
VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "reference_doc")
