"""Request and response schemas for the OrcaNet FastAPI service."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class TransferScoreRequest(BaseModel):
    """Request body for ``POST /api/v1/transfer/score``."""

    source_task_id: str
    target_task_id: str
    strategy: str = "feature"


class TransferRecommendRequest(BaseModel):
    """Request body for ``POST /api/v1/transfer/recommend``."""

    target_task_id: str
    query_description: str
    top_k: int = 3


class RetrieveRequest(BaseModel):
    """Request body for ``POST /api/v1/retrieve``."""

    task_id: str
    query_description: str | None = None
    filters: dict | None = None
    top_k: int = 10


class EmbedRequest(BaseModel):
    """Request body for ``POST /api/v1/cross-domain-embed``.

    Exactly one of ``task_id`` or ``statistical_features`` must be provided.
    """

    task_id: str | None = None
    statistical_features: list[float] | None = Field(default=None, min_length=25, max_length=25)
    description: str | None = None

    @model_validator(mode="after")
    def _require_exactly_one_source(self) -> "EmbedRequest":
        if (self.task_id is None) == (self.statistical_features is None):
            raise ValueError("Provide exactly one of task_id or statistical_features")
        return self


class ExplainRequest(BaseModel):
    """Request body for ``POST /api/v1/explain``."""

    source_task_id: str
    target_task_id: str
    strategy: str = "feature"


class EmbedResponse(BaseModel):
    """Response body for ``POST /api/v1/cross-domain-embed``."""

    embedding: list[float]


class ExplainResponse(BaseModel):
    """Response body for ``POST /api/v1/explain``."""

    explanation: str


class TransferValidateRequest(BaseModel):
    """Request body for ``POST /api/v1/transfer/validate``.

    The ``validate`` JSON key maps to the ``run_validation`` field to avoid
    shadowing :meth:`pydantic.BaseModel.validate`.
    """

    source_task_id: str
    target_task_id: str
    strategy: str = "feature"
    run_validation: bool = Field(default=True, alias="validate")

    model_config = {"populate_by_name": True}
