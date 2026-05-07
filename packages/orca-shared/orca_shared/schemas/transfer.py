from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class TransferMapping(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    mapping_id: UUID
    source_task_id: UUID
    target_task_id: UUID
    transfer_score: float
    transfer_type: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: datetime


class TransferScore(BaseModel):
    source_task_id: UUID
    target_task_id: UUID
    score: float


class TransferRecommendation(BaseModel):
    target_task_id: UUID
    recommended_sources: list[TransferMapping]
    top_score: float
