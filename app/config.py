"""
Central knobs for the chatbot.

═══════════════════════════════════════════════════════════════════════════
  >>> PLUG IN YOUR REFERENCE DOCUMENT HERE <<<
═══════════════════════════════════════════════════════════════════════════
  1. Drop your .docx into ./data/
  2. Either rename it to `reference.docx` OR override REFERENCE_DOC_PATH
     via the env var (docker-compose.yml already wires the data/ mount).
  3. Tweak CHUNK_SIZE / CHUNK_OVERLAP / TOP_K below if retrieval feels off.
  4. To force a re-ingest after changing chunking, delete ./storage/chroma.
═══════════════════════════════════════════════════════════════════════════
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


# ─── Gemini models (you can swap to gemini-1.5-flash for cheaper/faster) ─
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-1.5-pro")
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.4"))


# ─── On-disk locations (mounted volume in Docker) ───────────────────────
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "/app/storage"))
VECTOR_STORE_DIR = STORAGE_DIR / "chroma"
CHECKPOINT_DB = STORAGE_DIR / "checkpoints.sqlite"


# ─── Credentials (never hardcoded; injected via env) ────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
