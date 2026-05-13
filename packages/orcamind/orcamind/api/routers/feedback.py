"""Experiment feedback endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from orca_shared.registry.repository import ExperimentRepository, PerformanceRepository
from orca_shared.schemas.recommendation import FeedbackRequest

from ..deps import get_experiment_repo, get_perf_repo

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=dict)
async def submit_feedback(
    body: FeedbackRequest,
    exp_repo: ExperimentRepository = Depends(get_experiment_repo),
    perf_repo: PerformanceRepository = Depends(get_perf_repo),
) -> dict:
    experiment = await exp_repo.get_by_id(body.experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    await perf_repo.log_metric(
        experiment_id=body.experiment_id,
        name=body.metric_name,
        value=body.actual_metric,
        is_final=True,
    )
    return {"accepted": True}
