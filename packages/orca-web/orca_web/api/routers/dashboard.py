"""Dashboard aggregation endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from orca_web.api.deps import get_aggregator, get_current_user
from orca_web.models.user import User
from orca_web.services.aggregator import Aggregator

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/overview")
async def overview(
    _user: User = Depends(get_current_user),
    aggregator: Aggregator = Depends(get_aggregator),
) -> dict[str, Any]:
    return await aggregator.overview()


@router.get("/stats")
async def public_stats(
    aggregator: Aggregator = Depends(get_aggregator),
) -> dict[str, Any]:
    """Public stats for the landing page – no auth required."""
    return await aggregator.public_stats()
