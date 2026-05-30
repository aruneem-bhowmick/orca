"""LangChain tool for computing embedding similarity between two tasks."""

from __future__ import annotations

import json
import logging
import math
from uuid import UUID

import numpy as np
import torch

logger = logging.getLogger(__name__)

_embedder = None
_task_repository = None


def set_embedder(embedder) -> None:
    """Set the module-level cross-domain embedder used by ``embedding_similarity_tool``."""
    global _embedder
    _embedder = embedder


def set_task_repository(repo) -> None:
    """Set the module-level task repository used by ``embedding_similarity_tool``."""
    global _task_repository
    _task_repository = repo


def get_embedder():
    """Return the currently configured module-level embedder (may be ``None``)."""
    return _embedder


def get_task_repository():
    """Return the currently configured module-level task repository (may be ``None``)."""
    return _task_repository


def _task_to_feature_vector(task) -> np.ndarray:
    vec = np.zeros(25, dtype=np.float32)
    if task.n_samples is not None:
        vec[0] = math.log1p(float(task.n_samples))
    if task.n_features is not None:
        vec[1] = float(task.n_features)
    if task.n_classes is not None:
        vec[2] = float(task.n_classes)
    return vec


from langchain_core.tools import StructuredTool, tool


async def _run_embedding_similarity(
    task_id_a: str, task_id_b: str, *, embedder, task_repository
) -> str:
    if embedder is None or task_repository is None:
        return json.dumps({"error": "Embedder or task repository not configured"})
    try:
        task_a = await task_repository.get_by_id(UUID(task_id_a))
        task_b = await task_repository.get_by_id(UUID(task_id_b))
        if task_a is None or task_b is None:
            return json.dumps({"error": "One or both tasks not found"})

        vec_a = torch.from_numpy(_task_to_feature_vector(task_a)).unsqueeze(0)
        vec_b = torch.from_numpy(_task_to_feature_vector(task_b)).unsqueeze(0)

        emb_a = embedder.embed(vec_a).squeeze(0).detach()
        emb_b = embedder.embed(vec_b).squeeze(0).detach()

        emb_a = torch.nn.functional.normalize(emb_a, dim=0)
        emb_b = torch.nn.functional.normalize(emb_b, dim=0)
        similarity = float(torch.dot(emb_a, emb_b).clamp(-1.0, 1.0).item())
        return json.dumps({"similarity": similarity})
    except Exception as exc:
        logger.warning("embedding_similarity_tool failed: %s", exc)
        return json.dumps({"error": str(exc)})


@tool
async def embedding_similarity_tool(task_id_a: str, task_id_b: str) -> str:
    """Compute embedding similarity between two tasks. Returns a float 0-1."""
    return await _run_embedding_similarity(
        task_id_a, task_id_b, embedder=_embedder, task_repository=_task_repository
    )


def make_embedding_similarity_tool(embedder, task_repository) -> StructuredTool:
    """Return a new StructuredTool instance bound to the given embedder and repository."""

    async def _run(task_id_a: str, task_id_b: str) -> str:
        return await _run_embedding_similarity(
            task_id_a, task_id_b, embedder=embedder, task_repository=task_repository
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="embedding_similarity_tool",
        description=embedding_similarity_tool.description,
    )
