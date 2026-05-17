from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class SearchSpaceRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    search_space_id: UUID
    name: str | None = None
    definition: dict[str, Any]
    parent_id: UUID | None = None
    created_at: datetime
