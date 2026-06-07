"""
LangGraph wiring — the heart of "doesn't feel robotic".

State flow
──────────
State has two fields:

    messages       — Annotated with add_messages, so the model sees the
                     FULL conversation history on every turn. No node
                     ever overwrites or trims it.
    current_step   — int. 0 means solving hasn't started. N>=1 means the
                     model has delivered step N and the user owes a
                     confirmation. The MODEL READS THIS; it does not
                     decide it. We mutate it only when the model emits
                     STEP_DONE_MARKER (see below).

Graph topology (deliberately simple)
────────────────────────────────────
    START
      │
      ▼
    converse  ◀────────────┐
      │ (tool_calls?)      │
      ├──► tools ──────────┘   (retrieve_docs — facts only)
      │ (final reply)
      ▼
    pause  ◀── interrupt() suspends here, awaiting next user message;
      │       Command(resume=user_msg) restarts THIS node, appends the
      │       user message via add_messages, and edges back to converse.
      ▼
    converse  (loops forever until container shuts down)

There is ONE conversational node. Scope / diagnose / solve all happen
inside `converse`, driven by the shared persona prompt + `current_step`.
There are NO node-to-node handoffs between conversation phases — handoffs
are what make these bots feel like robots passing a baton.

`tools` and `pause` are not "phases"; `tools` is pure execution boilerplate
for retrieve_docs, and `pause` is the turn-taking mechanism. Neither
generates conversation text.

Where the interesting things happen:
  - retrieve_docs is called inside `tools` (the ToolNode), invoked when
    `converse` returns an AIMessage with tool_calls.
  - interrupt() is called inside `pause` to hand control back to the API.
  - Command(resume=...) restarts `pause` with the user's next message and
    flows naturally back into `converse` — we never re-invoke the graph
    from scratch with fresh input. The thread_id carries everything.
  - current_step advances ONLY in `converse`, ONLY when the model emits
    STEP_DONE_MARKER. The model never re-guesses where it is.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

from .config import (
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    LLM_SERVER_URL,
)
from .llm_client import RemoteChatModel
from .persona import STEP_DONE_MARKER, build_system_prompt
from .tools import retrieve_docs


# ─── State ─────────────────────────────────────────────────────────────
class ChatState(TypedDict):
    # add_messages reducer → full history every turn, nothing dropped.
    messages: Annotated[list, add_messages]
    # READ by the model via the system prompt; mutated only in converse()
    # when the model emits STEP_DONE_MARKER.
    current_step: int


# ─── Model (single instance, bound with the one tool) ──────────────────
# Lazily constructed on first use so module IMPORT doesn't require a key —
# the key is only validated when we actually try to generate. Cached after
# first call so we still only build the client once per process.
TOOLS = [retrieve_docs]

_llm_with_tools = None


def _get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        llm = RemoteChatModel(
            base_url=LLM_SERVER_URL,
            model=GENERATION_MODEL,
            temperature=GENERATION_TEMPERATURE,
        )
        _llm_with_tools = llm.bind_tools(TOOLS)
    return _llm_with_tools


# ─── The one conversational node ───────────────────────────────────────
def converse(state: ChatState) -> dict:
    """Generate one assistant turn.

    The same persona + current_step → focus hint is constructed each call.
    The model decides on its own (using the full message history) whether
    to call retrieve_docs, ask a diagnostic question, or deliver a step.
    """
    current_step = state.get("current_step", 0)
    system = SystemMessage(content=build_system_prompt(current_step))

    response = _get_llm_with_tools().invoke([system, *state["messages"]])
    update: dict = {"messages": [response]}

    # If the model wants a tool call, we don't touch the marker — tools
    # round-trip back here and the *next* converse() call will produce
    # the user-facing reply.
    has_tool_calls = bool(getattr(response, "tool_calls", None))
    if has_tool_calls:
        return update

    # Final user-facing reply: check for the bookkeeping marker.
    # Gemini may return content as a list of parts ([{"type":"text",...}]);
    # handle both shapes.
    raw = response.content
    if isinstance(raw, str):
        if STEP_DONE_MARKER in raw:
            response.content = raw.replace(STEP_DONE_MARKER, "").rstrip()
            update["current_step"] = current_step + 1
    elif isinstance(raw, list):
        marker_seen = False
        new_parts: list = []
        for p in raw:
            if isinstance(p, str):
                if STEP_DONE_MARKER in p:
                    marker_seen = True
                    p = p.replace(STEP_DONE_MARKER, "").rstrip()
                new_parts.append(p)
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                if STEP_DONE_MARKER in p["text"]:
                    marker_seen = True
                    p = {**p, "text": p["text"].replace(STEP_DONE_MARKER, "").rstrip()}
                new_parts.append(p)
            else:
                new_parts.append(p)
        if marker_seen:
            response.content = new_parts
            update["current_step"] = current_step + 1

    return update


# ─── Turn-taking node ──────────────────────────────────────────────────
def pause_for_user(state: ChatState) -> dict:
    """Hand control back to the caller and wait for the next user message.

    interrupt() suspends the graph here; the checkpointer persists state
    under the thread_id. The next /chat call resumes this node via
    Command(resume=user_msg) — the graph does NOT restart from START.
    """
    user_msg = interrupt({"awaiting": "user_message"})
    return {"messages": [{"role": "user", "content": user_msg}]}


# ─── Compile the graph with the supplied checkpointer ──────────────────
def build_graph(checkpointer):
    g = StateGraph(ChatState)

    g.add_node("converse", converse)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("pause", pause_for_user)

    g.add_edge(START, "converse")

    # tools_condition returns "tools" if the last message has tool_calls,
    # else the sentinel "__end__" — we redirect that to our pause node so
    # the graph waits for the user instead of terminating.
    g.add_conditional_edges(
        "converse",
        tools_condition,
        {"tools": "tools", "__end__": "pause"},
    )
    g.add_edge("tools", "converse")
    g.add_edge("pause", "converse")

    return g.compile(checkpointer=checkpointer)
