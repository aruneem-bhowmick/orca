"""Available model architectures endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from orca_shared.registry.models import Model as ModelORM
from orca_shared.schemas.model import ModelConfig

from ..deps import get_db

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelConfig])
async def list_models(
    limit: int = Query(default=100, ge=1, le=500),
    session: AsyncSession = Depends(get_db),
) -> list[ModelConfig]:
    result = await session.execute(select(ModelORM).limit(limit))
    return [ModelConfig.model_validate(row) for row in result.scalars()]
