"""
FastAPI entrypoint.

Startup (lifespan)
──────────────────
1. Build (or reload) the pgvector store — embedding the .docx if it
   hasn't been embedded yet. Done ONCE per process; never per request.
3. Open a PostgresSaver checkpointer against the postgres container, so
   conversation threads survive container restarts.
4. Compile the LangGraph with that checkpointer.

Per-request
───────────
POST /chat with { session_id, message }:
  - session_id maps 1:1 to a LangGraph thread_id.
  - If this thread has no prior state, we invoke the graph with initial
    state {messages:[user], current_step:0}.
  - If state exists (the graph paused at interrupt() last turn), we
    resume with Command(resume=message). We NEVER re-invoke from START
    with fresh input — that would restart the conversation.
  - We then pull the latest AIMessage from the persisted state and
    return its content as the reply.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from pathlib import Path as _Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
from psycopg_pool import ConnectionPool

from .config import POSTGRES_URL
from .graph import build_graph
from .ingest import build_or_load_vector_store


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("chatbot")


# Module-level singletons — built once at startup.
_graph = None
_pool: ConnectionPool | None = None


# PostgresSaver needs each connection in autocommit mode with
# prepare_threshold=0; these are the settings LangGraph documents.
_PG_CONN_KWARGS = {
    "autocommit": True,
    "prepare_threshold": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _pool

    log.info("Loading reference document and vector store …")
    build_or_load_vector_store()

    log.info("Opening PostgresSaver checkpointer at %s", POSTGRES_URL)
    # A pool (not a single conn) — FastAPI dispatches requests on
    # different threads via asyncio.to_thread, and a pool keeps each one
    # on its own connection.
    _pool = ConnectionPool(
        conninfo=POSTGRES_URL,
        max_size=20,
        kwargs=_PG_CONN_KWARGS,
    )
    checkpointer = PostgresSaver(_pool)
    # Idempotent — creates the checkpoint tables if missing.
    checkpointer.setup()

    _graph = build_graph(checkpointer)
    log.info("Graph compiled. Ready to chat.")

    try:
        yield
    finally:
        if _pool is not None:
            _pool.close()


app = FastAPI(title="RAG Support Chatbot", lifespan=lifespan)


# ─── Chat UI ────────────────────────────────────────────────────────────
# Serve a minimal self-contained chat page at GET / so the bot can be used
# from a browser at http://localhost:8000. The page hits the same /chat
# endpoint defined below — there's no second backend service.
_STATIC_DIR = _Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def chat_ui() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


class ChatIn(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1)


class ChatOut(BaseModel):
    reply: str
    current_step: int


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "graph_ready": _graph is not None}


@app.post("/chat", response_model=ChatOut)
async def chat(body: ChatIn) -> ChatOut:
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph not ready")

    # CRITICAL: same thread_id across the whole conversation so state
    # loads and the bot never restarts mid-thread.
    config = {"configurable": {"thread_id": body.session_id}}

    # Decide initial-invoke vs resume by checking persisted state.
    snapshot = _graph.get_state(config)
    is_new_thread = not snapshot.values

    def _run() -> None:
        # The graph runs to the next interrupt() and pauses; we don't
        # consume the return value — we read the latest AIMessage from
        # the persisted state below, which is the source of truth.
        if is_new_thread:
            _graph.invoke(
                {
                    "messages": [{"role": "user", "content": body.message}],
                    "current_step": 0,
                },
                config,
            )
        else:
            # Resume the paused thread — this is what keeps the
            # conversation continuous instead of restarting it.
            _graph.invoke(Command(resume=body.message), config)

    # LangGraph + PostgresSaver are sync; offload so we don't block the
    # event loop.
    await asyncio.to_thread(_run)

    final = _graph.get_state(config)
    messages = final.values.get("messages", [])
    current_step = int(final.values.get("current_step", 0))

    reply = _last_assistant_text(messages)
    if not reply:
        raise HTTPException(
            status_code=500,
            detail="No assistant reply produced (check logs).",
        )
    return ChatOut(reply=reply, current_step=current_step)


def _extract_text(content: Any) -> str:
    """Some chat models (notably Gemini via langchain-google-genai) return
    `AIMessage.content` as a list of structured parts instead of a plain
    string. Flatten that to the user-visible text. Tool-call payloads have
    no text and reduce to "".
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                txt = p.get("text") or p.get("content")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(parts)
    return ""


def _last_assistant_text(messages: list) -> str:
    """Pull the most recent assistant text from the message list.

    Messages may be LangChain BaseMessage objects (with .type/.content)
    or plain dicts ({"role": ..., "content": ...}); handle both. Skip
    messages whose content is empty (tool-call-only AIMessages).
    """
    for m in reversed(messages):
        role = getattr(m, "type", None) or (
            m.get("role") if isinstance(m, dict) else None
        )
        if role not in ("ai", "assistant"):
            continue
        raw = getattr(m, "content", None)
        if raw is None and isinstance(m, dict):
            raw = m.get("content")
        text = _extract_text(raw).strip()
        if text:
            return text
    return ""
