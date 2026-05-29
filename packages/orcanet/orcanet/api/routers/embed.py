"""Cross-domain embedding endpoint."""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

import numpy as np
import torch

from fastapi import APIRouter, Depends, HTTPException

from orca_shared.registry.repository import TaskRepository
from orcanet.api.deps import get_cross_domain_embedder, get_task_repo
from orcanet.api.schemas import EmbedRequest, EmbedResponse
from orcanet.embeddings.cross_domain import CrossDomainEmbedder
from orcanet.retrieval.retriever import task_to_feature_vector

logger = logging.getLogger("orcanet.api")

router = APIRouter(tags=["embed"])


@router.post("/cross-domain-embed", response_model=EmbedResponse)
async def cross_domain_embed(
    body: EmbedRequest,
    task_repo: TaskRepository = Depends(get_task_repo),
    embedder: CrossDomainEmbedder = Depends(get_cross_domain_embedder),
) -> EmbedResponse:
    """Return a 64-dim domain-invariant embedding for the given task."""
    if body.task_id is not None:
        try:
            task_uuid = UUID(body.task_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        task = await task_repo.get_by_id(task_uuid)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {body.task_id} not found")

        feature_vec = task_to_feature_vector(task)
    else:
        feature_vec = np.array(body.statistical_features, dtype=np.float32)

    input_tensor = torch.from_numpy(feature_vec).unsqueeze(0)
    embedding_tensor = await asyncio.to_thread(embedder.embed, input_tensor)
    return EmbedResponse(embedding=embedding_tensor.squeeze(0).tolist())
