from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator


class Embedding(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    embedding_id: UUID
    task_id: UUID | None = None
    embedding_type: str | None = None
    embedding_vector: list[float]
    dimension: int
    model_version: str | None = None
    created_at: datetime

    @model_validator(mode="after")
    def _validate_dimension(self) -> "Embedding":
        if self.dimension != len(self.embedding_vector):
            raise ValueError(
                f"dimension {self.dimension} does not match len(embedding_vector) {len(self.embedding_vector)}"
            )
        return self


class SimilarityResult(BaseModel):
    task_id: UUID
    score: float
    rank: int
