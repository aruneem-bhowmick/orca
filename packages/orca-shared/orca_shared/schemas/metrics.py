from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class MetricPoint(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    value: float
    step: int | None = None
    is_final: bool = False


class PerformanceMetrics(BaseModel):
    experiment_id: UUID
    final_metrics: dict[str, float]
    best_epoch: int | None = None


class PerformanceSummary(BaseModel):
    task_name: str
    architecture: str
    mean_accuracy: float
