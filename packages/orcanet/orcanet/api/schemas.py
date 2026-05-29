"""Request and response schemas for the OrcaNet FastAPI service."""

from __future__ import annotations

from pydantic import BaseModel, model_validator


class TransferScoreRequest(BaseModel):
    source_task_id: str
    target_task_id: str
    strategy: str = "feature"


class TransferRecommendRequest(BaseModel):
    target_task_id: str
    query_description: str
    top_k: int = 3


class RetrieveRequest(BaseModel):
    task_id: str
    query_description: str | None = None
    filters: dict | None = None
    top_k: int = 10


class EmbedRequest(BaseModel):
    task_id: str | None = None
    statistical_features: list[float] | None = None
    description: str | None = None

    @model_validator(mode="after")
    def _require_at_least_one_source(self) -> "EmbedRequest":
        if self.task_id is None and self.statistical_features is None:
            raise ValueError("Provide either task_id or statistical_features")
        return self


class ExplainRequest(BaseModel):
    source_task_id: str
    target_task_id: str
    strategy: str = "feature"
