"""Model recommendation, performance prediction, and task similarity endpoints."""

from __future__ import annotations

from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from orca_shared.registry.models import Model as ModelORM
from orca_shared.schemas.embedding import SimilarityResult
from orca_shared.schemas.model import ModelConfig
from orca_shared.schemas.recommendation import ModelRecommendation, RecommendationRequest
from orcamind.embedders.similarity import FaissIndex
from orcamind.selectors.nearest_neighbor import NearestNeighborSelector
from orcamind.selectors.predictor import PerformancePredictor

from ..deps import get_db, get_faiss_index, get_nn_selector, get_predictor

router = APIRouter(tags=["recommend"])


class PredictRequest(BaseModel):
    task_embedding: list[float]
    model_id: UUID


class PredictResponse(BaseModel):
    model_id: UUID
    predicted_score: float
    confidence: float


class SimilarTasksRequest(BaseModel):
    task_embedding: list[float]
    top_k: int = Field(default=5, ge=1, le=100)


async def _load_model_configs(
    session: AsyncSession, limit: int = 500
) -> list[ModelConfig]:
    result = await session.execute(select(ModelORM).limit(limit))
    return [ModelConfig.model_validate(row) for row in result.scalars()]


@router.post("/recommend-model", response_model=list[ModelRecommendation])
async def recommend_model(
    body: RecommendationRequest,
    selector: NearestNeighborSelector = Depends(get_nn_selector),
    session: AsyncSession = Depends(get_db),
) -> list[ModelRecommendation]:
    model_configs = await _load_model_configs(session)
    embedding = np.array(body.task_embedding, dtype=np.float64)
    try:
        return selector.recommend(embedding, model_configs, top_k=body.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/predict-performance", response_model=PredictResponse)
async def predict_performance(
    body: PredictRequest,
    predictor: PerformancePredictor = Depends(get_predictor),
    session: AsyncSession = Depends(get_db),
) -> PredictResponse:
    result = await session.execute(
        select(ModelORM).where(ModelORM.model_id == body.model_id)
    )
    model_row = result.scalar_one_or_none()
    if model_row is None:
        raise HTTPException(status_code=404, detail="Model not found")

    model_config = ModelConfig.model_validate(model_row)
    embedding = np.array(body.task_embedding, dtype=np.float64)
    try:
        score, confidence = predictor.predict_with_confidence(embedding, model_config)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictResponse(
        model_id=body.model_id,
        predicted_score=score,
        confidence=confidence,
    )


@router.post("/similar-tasks", response_model=list[SimilarityResult])
async def similar_tasks(
    body: SimilarTasksRequest,
    faiss_index: FaissIndex = Depends(get_faiss_index),
) -> list[SimilarityResult]:
    embedding = np.array(body.task_embedding, dtype=np.float32)
    raw = faiss_index.search(embedding, k=body.top_k)
    return [
        SimilarityResult(task_id=UUID(tid), score=score, rank=rank)
        for rank, (tid, score) in enumerate(raw, start=1)
    ]
