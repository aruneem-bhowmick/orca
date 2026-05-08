from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


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
    # populate_by_name lets callers use "metadata" directly while
    # from_attributes mode will resolve via "task_metadata" on the ORM row
    # (avoiding the DeclarativeBase.metadata class-attribute collision).
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    task_id: UUID
    name: str
    domain: str | None = None
    task_type: str
    n_samples: int | None = None
    n_features: int | None = None
    n_classes: int | None = None
    dataset_uri: str | None = None
    metadata: dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("task_metadata", "metadata"),
    )
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
