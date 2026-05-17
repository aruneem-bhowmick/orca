"""Sweep management endpoints — trigger and inspect Prefect-backed sweeps."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from ..deps import get_sweeps_store

logger = logging.getLogger("orcalab.api")

router = APIRouter(prefix="/sweeps", tags=["sweeps"])


class StartSweepRequest(BaseModel):
    task_id: str
    n_trials: int = Field(default=50, ge=1)
    use_orcamind: bool = True
    search_space: dict[str, Any] | None = None


class SweepStatus(BaseModel):
    sweep_id: str
    task_id: str
    n_trials_total: int
    n_completed: int
    n_failed: int
    best_result: dict[str, Any] | None = None
    flow_run_id: str | None = None


class TrialResult(BaseModel):
    trial_id: str
    objective: float
    params: dict[str, Any]
    status: str = "completed"


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def start_sweep(
    body: StartSweepRequest,
    request: Request,
    sweeps: dict = Depends(get_sweeps_store),
) -> dict[str, str]:
    sweep_id = str(uuid.uuid4())
    prefect_url = os.environ.get("PREFECT_API_URL", "")
    flow_run_id: str | None = None

    if prefect_url:
        deployment_name = "meta_informed_sweep/default"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{prefect_url}/deployments/name/{deployment_name}/create_flow_run",
                    json={
                        "name": f"sweep-{sweep_id[:8]}",
                        "parameters": {
                            "task_id": body.task_id,
                            "n_trials": body.n_trials,
                            "use_orcamind": body.use_orcamind,
                            "search_space": body.search_space,
                        },
                    },
                )
            if resp.status_code in (200, 201):
                try:
                    flow_run_id = resp.json().get("id")
                except ValueError:
                    logger.warning(
                        "Prefect returned non-JSON success response: status=%s",
                        resp.status_code,
                    )
        except httpx.HTTPError as exc:
            logger.warning("Failed to trigger Prefect flow: %s", exc)

    sweeps[sweep_id] = {
        "sweep_id": sweep_id,
        "task_id": body.task_id,
        "n_trials_total": body.n_trials,
        "n_completed": 0,
        "n_failed": 0,
        "flow_run_id": flow_run_id,
        "search_space": body.search_space,
        "results": [],
    }

    return {"sweep_id": sweep_id}


@router.get("/{sweep_id}", response_model=SweepStatus)
async def get_sweep_status(
    sweep_id: str,
    sweeps: dict = Depends(get_sweeps_store),
) -> SweepStatus:
    sweep = sweeps.get(sweep_id)
    if sweep is None:
        raise HTTPException(status_code=404, detail="Sweep not found")

    best_result: dict[str, Any] | None = None
    if sweep["results"]:
        best = max(sweep["results"], key=lambda r: r.get("objective", 0.0))
        best_result = best

    return SweepStatus(
        sweep_id=sweep_id,
        task_id=sweep["task_id"],
        n_trials_total=sweep["n_trials_total"],
        n_completed=sweep["n_completed"],
        n_failed=sweep["n_failed"],
        best_result=best_result,
        flow_run_id=sweep.get("flow_run_id"),
    )


@router.get("/{sweep_id}/results", response_model=list[TrialResult])
async def get_sweep_results(
    sweep_id: str,
    sweeps: dict = Depends(get_sweeps_store),
) -> list[TrialResult]:
    sweep = sweeps.get(sweep_id)
    if sweep is None:
        raise HTTPException(status_code=404, detail="Sweep not found")

    sorted_results = sorted(
        sweep["results"], key=lambda r: r.get("objective", 0.0), reverse=True
    )
    return [TrialResult(**r) for r in sorted_results]
