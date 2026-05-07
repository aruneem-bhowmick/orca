from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class TaskCreate(BaseModel):
    name: str
    domain: str | None = None
    task_type: str
    n_samples: int | None = None
    n_features: int | None = None
    n_classes: int | None = None
    dataset_uri: str | None = None
    metadata: dict[str, Any] | None = None


class Task(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: UUID
    name: str
    domain: str | None = None
    task_type: str
    n_samples: int | None = None
    n_features: int | None = None
    n_classes: int | None = None
    dataset_uri: str | None = None
    metadata: dict[str, Any] | None = None
    embedding_id: UUID | None = None
    created_at: datetime
    updated_at: datetime


class TaskSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: UUID
    name: str
    domain: str | None = None
    task_type: str


class DatasetSummary(BaseModel):
    dataset_uri: str
    n_samples: int
    n_features: int
    n_classes: int | None = None
