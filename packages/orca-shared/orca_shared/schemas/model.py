from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ModelConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_id: UUID
    name: str
    architecture: str | None = None
    config: dict[str, Any]
    parameter_count: int | None = None
    flops: int | None = None


class ModelSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_id: UUID
    name: str
    architecture: str | None = None
