"""Async meta-adaptation job endpoint."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from orca_shared.registry.models import Model as ModelORM
from orca_shared.registry.repository import ExperimentRepository, TaskRepository

from ..deps import get_db, get_experiment_repo, get_task_repo

router = APIRouter(prefix="/adapt", tags=["adapt"])
logger = logging.getLogger("orcamind.api.adapt")


class AdaptRequest(BaseModel):
    task_id: UUID
    model_id: UUID
    training_config: dict[str, Any] | None = None


async def _run_adaptation(engine: AsyncEngine, experiment_id: UUID) -> None:
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as session:
            async with session.begin():
                await ExperimentRepository(session).update_status(experiment_id, "running")
        logger.info("Adaptation job %s started", experiment_id)
        # Placeholder: real meta-adaptation invokes WarmStartTransfer here
        async with factory() as session:
            async with session.begin():
                await ExperimentRepository(session).update_status(experiment_id, "completed")
        logger.info("Adaptation job %s completed", experiment_id)
    except Exception as exc:
        logger.exception("Adaptation job %s failed: %s", experiment_id, exc)
        try:
            async with factory() as session:
                async with session.begin():
                    await ExperimentRepository(session).update_status(experiment_id, "failed")
        except Exception:
            pass


@router.post("", response_model=dict)
async def start_adaptation(
    body: AdaptRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    task_repo: TaskRepository = Depends(get_task_repo),
    exp_repo: ExperimentRepository = Depends(get_experiment_repo),
    session: AsyncSession = Depends(get_db),
) -> dict:
    task = await task_repo.get_by_id(body.task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    result = await session.execute(
        select(ModelORM).where(ModelORM.model_id == body.model_id)
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Model not found")

    experiment = await exp_repo.create(
        task_id=body.task_id,
        model_id=body.model_id,
        training_config=body.training_config,
    )

    background_tasks.add_task(
        _run_adaptation, request.app.state.db_engine, experiment.experiment_id
    )
    return {"job_id": str(experiment.experiment_id)}
