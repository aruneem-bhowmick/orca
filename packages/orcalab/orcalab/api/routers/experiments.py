"""Experiment CRUD endpoints and live WebSocket stream."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel

from orca_shared.registry.repository import ExperimentRepository
from orca_shared.schemas.training import ExperimentResult
from orcalab.experiments.lifecycle import ExperimentLifecycle, ExperimentStatus, InvalidTransitionError

from ..deps import get_experiment_repo

logger = logging.getLogger("orcalab.api")

router = APIRouter(prefix="/experiments", tags=["experiments"])

_TERMINAL = {ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.CANCELLED}
_CANCELLABLE = {ExperimentStatus.PENDING, ExperimentStatus.QUEUED, ExperimentStatus.RUNNING}


class CreateExperimentRequest(BaseModel):
    task_id: UUID | None = None
    model_id: UUID | None = None
    training_config: dict[str, Any] | None = None
    created_by: str | None = None


@router.post("", response_model=ExperimentResult, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    body: CreateExperimentRequest,
    repo: ExperimentRepository = Depends(get_experiment_repo),
) -> ExperimentResult:
    return await repo.create(
        task_id=body.task_id,
        model_id=body.model_id,
        training_config=body.training_config,
        created_by=body.created_by,
    )


@router.get("", response_model=list[ExperimentResult])
async def list_experiments(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    repo: ExperimentRepository = Depends(get_experiment_repo),
) -> list[ExperimentResult]:
    return await repo.list_all(limit=limit, offset=offset)


@router.get("/{experiment_id}", response_model=ExperimentResult)
async def get_experiment(
    experiment_id: UUID,
    repo: ExperimentRepository = Depends(get_experiment_repo),
) -> ExperimentResult:
    experiment = await repo.get_by_id(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.delete("/{experiment_id}", response_model=ExperimentResult)
async def cancel_experiment(
    experiment_id: UUID,
    repo: ExperimentRepository = Depends(get_experiment_repo),
) -> ExperimentResult:
    experiment = await repo.get_by_id(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    current_status = ExperimentStatus(experiment.status)
    if current_status not in _CANCELLABLE:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel experiment with status {experiment.status!r}",
        )

    from orcalab.experiments.experiment import Experiment as OrcaExperiment

    orca_exp = OrcaExperiment(**experiment.model_dump())
    lifecycle = ExperimentLifecycle(orca_exp, repo)
    try:
        await lifecycle.transition(ExperimentStatus.CANCELLED, reason="api-delete")
    except InvalidTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    updated = await repo.get_by_id(experiment_id)
    assert updated is not None
    return updated


@router.websocket("/{experiment_id}/live")
async def experiment_live(websocket: WebSocket, experiment_id: UUID) -> None:
    await websocket.accept()
    try:
        while True:
            async with websocket.app.state.db_sessionmaker() as session:  # type: ignore[attr-defined]
                live_repo = ExperimentRepository(session)
                experiment = await live_repo.get_by_id(experiment_id)

            if experiment is None:
                await websocket.send_json({"error": "experiment not found"})
                break

            metrics: dict[str, Any] = experiment.metrics or {}
            await websocket.send_json(
                {
                    "experiment_id": str(experiment_id),
                    "status": experiment.status,
                    "epoch": metrics.get("epoch"),
                    "loss": metrics.get("loss"),
                    "metrics": metrics,
                }
            )

            if ExperimentStatus(experiment.status) in _TERMINAL:
                break

            await asyncio.sleep(2)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected for experiment %s", experiment_id)
    except Exception as exc:
        logger.warning("WebSocket error for experiment %s: %s", experiment_id, exc)
