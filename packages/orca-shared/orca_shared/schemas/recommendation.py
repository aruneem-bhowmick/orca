from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    task_embedding: list[float]
    domain: str | None = None
    task_type: str | None = None
    top_k: int = Field(default=5, ge=1)


class ModelRecommendation(BaseModel):
    task_id: UUID
    model_id: UUID
    architecture: str | None = None
    predicted_score: float
    confidence: float | None = None
    reasoning: str | None = None


class FeedbackRequest(BaseModel):
    experiment_id: UUID
    actual_metric: float
    metric_name: str
    params: dict[str, Any] | None = None
