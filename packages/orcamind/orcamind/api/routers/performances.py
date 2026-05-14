"""Performance summary endpoint for dashboard heatmap."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from orca_shared.registry.repository import PerformanceRepository
from orca_shared.schemas.metrics import PerformanceSummary

from ..deps import get_perf_repo

router = APIRouter(prefix="/performances", tags=["performances"])


@router.get("", response_model=list[PerformanceSummary])
async def list_performances(
    metric_name: str = "accuracy",
    perf_repo: PerformanceRepository = Depends(get_perf_repo),
) -> list[PerformanceSummary]:
    """Return mean metric values grouped by task and architecture.

    Args:
        metric_name: Name of the metric to aggregate (default: ``"accuracy"``).

    Returns:
        List of :class:`PerformanceSummary` objects, one per (task, architecture) pair.
    """
    return await perf_repo.list_all_with_context(metric_name=metric_name)
