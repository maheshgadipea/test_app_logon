"""
Gemini LLM server.

A thin FastAPI proxy in front of Google Gemini. The chatbot container
talks to this service over HTTP for every model interaction:

    POST /v1/generate  →  gemini-2.5-flash (chat + tool calling)
    POST /v1/embed     →  text-embedding-004

The Google API key lives ONLY in this container's environment. The
chatbot container never imports `langchain-google-genai` and never holds
the key.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("llm-server")


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
DEFAULT_GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")


def _require_key() -> str:
    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY (or GEMINI_API_KEY) is not set in llm-server env.",
        )
    return GOOGLE_API_KEY


# ─── Wire schema ────────────────────────────────────────────────────────
class MessageIn(BaseModel):
    role: str  # system | user | ai | tool
    content: Any = ""
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ToolDef(BaseModel):
    name: str
    description: str = ""
    parameters: dict = Field(default_factory=dict)


class GenerateIn(BaseModel):
    messages: list[MessageIn]
    tools: list[ToolDef] | None = None
    model: str | None = None
    temperature: float = 0.4


class GenerateOut(BaseModel):
    content: Any
    tool_calls: list[dict] = Field(default_factory=list)


class EmbedIn(BaseModel):
    texts: list[str]
    model: str | None = None


class EmbedOut(BaseModel):
    embeddings: list[list[float]]


# ─── Model caches (one client per (model, temperature/tools) combo) ─────
_chat_cache: dict[tuple, Any] = {}
_embed_cache: dict[str, GoogleGenerativeAIEmbeddings] = {}


def _get_chat_llm(model: str, temperature: float, tools: list[ToolDef] | None):
    tool_key = tuple((t.name, t.description) for t in (tools or []))
    key = (model, temperature, tool_key)
    if key in _chat_cache:
        return _chat_cache[key]

    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=_require_key(),
        temperature=temperature,
    )
    if tools:
        # Pass tools as OpenAI-style function dicts; langchain-google-genai
        # translates these into Gemini's function-calling schema.
        tool_dicts = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters or {"type": "object", "properties": {}},
                },
            }
            for t in tools
        ]
        llm = llm.bind_tools(tool_dicts)

    _chat_cache[key] = llm
    return llm


def _get_embed_client(model: str) -> GoogleGenerativeAIEmbeddings:
    if model not in _embed_cache:
        _embed_cache[model] = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=_require_key(),
        )
    return _embed_cache[model]


def _to_lc_messages(messages: list[MessageIn]):
    out = []
    for m in messages:
        role = m.role.lower()
        content = m.content if m.content is not None else ""
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role in ("user", "human"):
            out.append(HumanMessage(content=content))
        elif role in ("ai", "assistant"):
            kwargs: dict[str, Any] = {}
            if m.tool_calls:
                kwargs["tool_calls"] = [
                    {
                        "id": tc.get("id") or f"call_{i}",
                        "name": tc["name"],
                        "args": tc.get("args", {}),
                        "type": "tool_call",
                    }
                    for i, tc in enumerate(m.tool_calls)
                ]
            out.append(AIMessage(content=content, **kwargs))
        elif role == "tool":
            out.append(
                ToolMessage(
                    content=content if isinstance(content, str) else str(content),
                    tool_call_id=m.tool_call_id or "",
                    name=m.name,
                )
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown message role: {m.role!r}"
            )
    return out


# ─── App ────────────────────────────────────────────────────────────────
app = FastAPI(title="Gemini LLM Server")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "has_key": bool(GOOGLE_API_KEY),
        "generation_model": DEFAULT_GENERATION_MODEL,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
    }


@app.post("/v1/generate", response_model=GenerateOut)
def generate(body: GenerateIn) -> GenerateOut:
    model = body.model or DEFAULT_GENERATION_MODEL
    llm = _get_chat_llm(model, body.temperature, body.tools)

    lc_messages = _to_lc_messages(body.messages)
    response = llm.invoke(lc_messages)

    tool_calls = []
    for tc in getattr(response, "tool_calls", None) or []:
        tool_calls.append(
            {
                "id": tc.get("id"),
                "name": tc.get("name"),
                "args": tc.get("args", {}),
            }
        )

    return GenerateOut(content=response.content, tool_calls=tool_calls)


@app.post("/v1/embed", response_model=EmbedOut)
def embed(body: EmbedIn) -> EmbedOut:
    if not body.texts:
        return EmbedOut(embeddings=[])
    model = body.model or DEFAULT_EMBEDDING_MODEL
    client = _get_embed_client(model)
    vectors = client.embed_documents(body.texts)
    return EmbedOut(embeddings=vectors)
