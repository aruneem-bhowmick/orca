from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TrainingConfig(BaseModel):
    batch_size: int = Field(default=32, gt=0)
    lr: float = Field(default=1e-3, gt=0)
    epochs: int = Field(default=10, gt=0)
    optimizer: str = "adam"
    scheduler: str | None = None
    extra: dict[str, Any] | None = None


class ExperimentResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    experiment_id: UUID
    task_id: UUID | None = None
    model_id: UUID | None = None
    status: str
    mlflow_run_id: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metrics: dict[str, float] | None = None
