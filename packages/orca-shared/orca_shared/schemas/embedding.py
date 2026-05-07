from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class Embedding(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    embedding_id: UUID
    task_id: UUID | None = None
    embedding_type: str | None = None
    embedding_vector: list[float]
    dimension: int
    model_version: str | None = None
    created_at: datetime


class SimilarityResult(BaseModel):
    task_id: UUID
    score: float
    rank: int
