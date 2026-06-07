"""
LangGraph tools.

Only ONE tool is exposed to the model: `retrieve_docs`. It returns raw
chunks from the reference document — facts only. The model COMPOSES the
diagnostic questions and the solution itself, in its own voice, using
those chunks. We deliberately do not have a "give_solution" tool, because
a pre-baked solution-fetching tool is what makes RAG bots feel robotic.
"""

from __future__ import annotations

from langchain_core.tools import tool

from .config import TOP_K
from .ingest import get_vector_store


@tool
def retrieve_docs(query: str) -> str:
    """Retrieve passages from the reference document relevant to `query`.

    Use this whenever you need facts from the document — to decide if a
    question is in scope, to shape a diagnostic question, or to compose
    the next solution step. The returned chunks are your only source of
    truth. Do NOT invent facts that aren't in them.
    """
    results = get_vector_store().similarity_search(query, k=TOP_K)
    if not results:
        return "(no relevant passages found in the reference document)"
    return "\n\n---\n\n".join(
        f"[chunk {i + 1}]\n{d.page_content}" for i, d in enumerate(results)
    )
