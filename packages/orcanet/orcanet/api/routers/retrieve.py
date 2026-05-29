"""Task retrieval endpoint."""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from orca_shared.registry.repository import TaskRepository
from orca_shared.schemas.embedding import SimilarityResult
from orcanet.api.deps import get_hybrid_retriever, get_task_repo
from orcanet.api.schemas import RetrieveRequest
from orcanet.retrieval.retriever import HybridRetriever

logger = logging.getLogger("orcanet.api")

router = APIRouter(tags=["retrieve"])


@router.post("/retrieve", response_model=list[SimilarityResult])
async def retrieve_similar_tasks(
    body: RetrieveRequest,
    task_repo: TaskRepository = Depends(get_task_repo),
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
) -> list[SimilarityResult]:
    """Return the top-k most similar tasks to the query task."""
    try:
        task_uuid = UUID(body.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    query_task = await task_repo.get_by_id(task_uuid)
    if query_task is None:
        raise HTTPException(status_code=404, detail=f"Task {body.task_id} not found")

    if body.query_description:
        raw = await retriever.retrieve_with_expanded_queries(
            body.query_description, query_task, task_repository=task_repo
        )
    else:
        raw = await retriever.retrieve(query_task, filters=body.filters, task_repository=task_repo)

    results = raw[: body.top_k]
    return [
        SimilarityResult(task_id=task.task_id, score=score, rank=rank + 1)
        for rank, (task, score, _) in enumerate(results)
    ]
