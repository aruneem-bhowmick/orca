"""LangChain tool for retrieving similar ML tasks from the registry."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from uuid import uuid4

from langchain_core.tools import StructuredTool, tool

logger = logging.getLogger(__name__)

_retriever = None


def set_retriever(retriever) -> None:
    """Set the module-level hybrid retriever used by ``task_retrieval_tool``."""
    global _retriever
    _retriever = retriever


def get_retriever():
    """Return the currently configured module-level retriever (may be ``None``)."""
    return _retriever


async def _run_task_retrieval(query: str, filters: str = "{}", *, retriever) -> str:
    """Run the hybrid retriever for *query*, apply optional metadata *filters*, and return JSON."""
    if retriever is None:
        return json.dumps({"error": "Task retriever not configured"})
    try:
        parsed_filters = json.loads(filters) if filters.strip() not in ("", "{}") else None
        from orca_shared.schemas.task import Task

        query_task = Task(
            task_id=uuid4(),
            name=query,
            domain="unknown",
            task_type="classification",
            n_samples=None,
            n_features=None,
            n_classes=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        results = await retriever.retrieve_with_expanded_queries(query, query_task)
        if parsed_filters:
            results = [
                (t, s, r)
                for t, s, r in results
                if all(getattr(t, k, None) == v for k, v in parsed_filters.items())
            ]
        return json.dumps(
            [
                {"task_id": str(t.task_id), "name": t.name, "score": s, "reason": r}
                for t, s, r in results
            ]
        )
    except Exception as exc:
        logger.warning("task_retrieval_tool failed: %s", exc)
        return json.dumps({"error": str(exc)})


@tool
async def task_retrieval_tool(query: str, filters: str = "{}") -> str:
    """Retrieve similar ML tasks from the registry. Returns JSON list of tasks with similarity scores."""
    return await _run_task_retrieval(query, filters, retriever=_retriever)


def make_task_retrieval_tool(retriever) -> StructuredTool:
    """Return a new StructuredTool instance bound to the given retriever."""

    async def _run(query: str, filters: str = "{}") -> str:
        """Delegate to ``_run_task_retrieval`` with the captured retriever closure."""
        return await _run_task_retrieval(query, filters, retriever=retriever)

    return StructuredTool.from_function(
        coroutine=_run,
        name="task_retrieval_tool",
        description=task_retrieval_tool.description,
    )
