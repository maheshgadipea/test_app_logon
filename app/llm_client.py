"""
HTTP client wrappers that make the remote llm-server look like local
LangChain components.

- `RemoteChatModel`  → drop-in for `ChatGoogleGenerativeAI`.
                      `.bind_tools(tools)` + `.invoke(messages)` work the
                      same way LangGraph expects.
- `RemoteEmbeddings` → drop-in for `GoogleGenerativeAIEmbeddings`. Plugs
                      straight into Chroma.

Both classes know nothing about Gemini, the API key, or the Google SDK;
they just POST JSON to the llm-server. That keeps the chatbot container
free of any Google credentials.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field


# ─── Serialization helpers ──────────────────────────────────────────────
def _content_to_json(content: Any) -> Any:
    """LangChain message content may be a string OR a list of structured
    parts (Gemini does this). Either is JSON-serializable as-is; anything
    else is coerced to str."""
    if isinstance(content, (str, list, dict)):
        return content
    return str(content) if content is not None else ""


def _message_to_dict(m: BaseMessage) -> dict:
    if isinstance(m, SystemMessage):
        return {"role": "system", "content": _content_to_json(m.content)}
    if isinstance(m, HumanMessage):
        return {"role": "user", "content": _content_to_json(m.content)}
    if isinstance(m, AIMessage):
        d: dict[str, Any] = {"role": "ai", "content": _content_to_json(m.content)}
        if m.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.get("id"),
                    "name": tc["name"],
                    "args": tc.get("args", {}),
                }
                for tc in m.tool_calls
            ]
        return d
    if isinstance(m, ToolMessage):
        return {
            "role": "tool",
            "content": _content_to_json(m.content),
            "tool_call_id": m.tool_call_id,
            "name": getattr(m, "name", None),
        }
    # Plain dicts pass through (LangGraph occasionally feeds these in).
    if isinstance(m, dict):
        return m
    return {"role": "user", "content": str(m)}


def _tool_to_schema(tool: Any) -> dict:
    """Reduce any tool input (BaseTool, Pydantic, dict, fn) to the simple
    {name, description, parameters} dict the llm-server expects."""
    if isinstance(tool, dict):
        fn = tool.get("function", tool)
        return {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        }
    schema = convert_to_openai_tool(tool)
    fn = schema["function"]
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "parameters": fn.get("parameters", {}),
    }


# ─── Chat model ─────────────────────────────────────────────────────────
class RemoteChatModel(BaseChatModel):
    """Calls llm-server's POST /v1/generate.

    Implements just enough of `BaseChatModel` for LangGraph: `_generate`
    returns a `ChatResult` whose only message is an `AIMessage` with
    `tool_calls` populated when Gemini decides to call a tool. LangGraph's
    `ToolNode` + `tools_condition` work unmodified.
    """

    base_url: str
    model: Optional[str] = None
    temperature: float = 0.4
    request_timeout: float = 120.0
    tools_schema: Optional[list[dict]] = Field(default=None)

    @property
    def _llm_type(self) -> str:
        return "remote-gemini"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "RemoteChatModel":
        schemas = [_tool_to_schema(t) for t in tools]
        return self.model_copy(update={"tools_schema": schemas})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "messages": [_message_to_dict(m) for m in messages],
            "temperature": self.temperature,
        }
        if self.model:
            payload["model"] = self.model
        if self.tools_schema:
            payload["tools"] = self.tools_schema

        resp = requests.post(
            f"{self.base_url.rstrip('/')}/v1/generate",
            json=payload,
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        tool_calls = []
        for tc in data.get("tool_calls") or []:
            tool_calls.append(
                {
                    "id": tc.get("id"),
                    "name": tc["name"],
                    "args": tc.get("args", {}),
                    "type": "tool_call",
                }
            )

        ai = AIMessage(content=data.get("content", ""), tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=ai)])


# ─── Embeddings ─────────────────────────────────────────────────────────
class RemoteEmbeddings(Embeddings):
    """Calls llm-server's POST /v1/embed. Plugs straight into Chroma."""

    def __init__(
        self,
        base_url: str,
        model: Optional[str] = None,
        request_timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.request_timeout = request_timeout

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, Any] = {"texts": texts}
        if self.model:
            payload["model"] = self.model
        resp = requests.post(
            f"{self.base_url}/v1/embed",
            json=payload,
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]
