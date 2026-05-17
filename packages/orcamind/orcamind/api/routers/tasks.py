"""Task registry endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from orca_shared.registry.repository import EmbeddingRepository, TaskRepository
from orca_shared.schemas.embedding import Embedding
from orca_shared.schemas.task import Task, TaskSummary

from ..deps import get_embedding_repo, get_task_repo

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}/embedding", response_model=Embedding)
async def get_task_embedding(
    task_id: UUID,
    task_repo: TaskRepository = Depends(get_task_repo),
    emb_repo: EmbeddingRepository = Depends(get_embedding_repo),
) -> Embedding:
    task = await task_repo.get_by_id(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.embedding_id is None:
        raise HTTPException(status_code=404, detail="Task has no embedding")
    embedding = await emb_repo.get_by_id(task.embedding_id)
    if embedding is None:
        raise HTTPException(status_code=404, detail="Embedding not found")
    return embedding


class EmbedTaskRequest(BaseModel):
    task_id: UUID
    embedding_vector: list[float]
    embedding_type: str = "statistical"
    model_version: str = "v1"


@router.get("", response_model=list[TaskSummary])
async def list_tasks(
    domain: str | None = Query(default=None),
    task_type: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    repo: TaskRepository = Depends(get_task_repo),
) -> list[TaskSummary]:
    if domain is not None and task_type is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Cannot filter by both domain and task_type simultaneously",
        )
    if domain is not None:
        return await repo.list_by_domain(domain, limit=limit, offset=offset)
    if task_type is not None:
        return await repo.list_by_type(task_type, limit=limit, offset=offset)
    return await repo.list_all(limit=limit, offset=offset)


@router.get("/{task_id}", response_model=Task)
async def get_task(
    task_id: UUID,
    repo: TaskRepository = Depends(get_task_repo),
) -> Task:
    task = await repo.get_by_id(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.post("/embed", response_model=Embedding)
async def embed_task(
    body: EmbedTaskRequest,
    task_repo: TaskRepository = Depends(get_task_repo),
    emb_repo: EmbeddingRepository = Depends(get_embedding_repo),
) -> Embedding:
    task = await task_repo.get_by_id(body.task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    embedding = await emb_repo.create(
        task_id=body.task_id,
        embedding_vector=body.embedding_vector,
        embedding_type=body.embedding_type,
        model_version=body.model_version,
    )
    await task_repo.update_embedding(body.task_id, embedding.embedding_id)
    return embedding
