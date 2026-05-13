"""Async meta-adaptation job endpoint."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from orca_shared.registry.repository import ExperimentRepository, TaskRepository
from orca_shared.registry.models import Model as ModelORM
from sqlalchemy import select

from ..deps import get_experiment_repo, get_task_repo

router = APIRouter(prefix="/adapt", tags=["adapt"])
logger = logging.getLogger("orcamind.api.adapt")


class AdaptRequest(BaseModel):
    task_id: UUID
    model_id: UUID
    training_config: dict[str, Any] | None = None


async def _run_adaptation(engine: AsyncEngine, experiment_id: UUID) -> None:
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        async with session.begin():
            repo = ExperimentRepository(session)
            await repo.update_status(experiment_id, "running")

    try:
        # Placeholder: real meta-adaptation would invoke WarmStartTransfer here
        logger.info("Adaptation job %s started", experiment_id)
        async with factory() as session:
            async with session.begin():
                repo = ExperimentRepository(session)
                await repo.update_status(experiment_id, "completed")
        logger.info("Adaptation job %s completed", experiment_id)
    except Exception as exc:
        logger.exception("Adaptation job %s failed: %s", experiment_id, exc)
        async with factory() as session:
            async with session.begin():
                repo = ExperimentRepository(session)
                await repo.update_status(experiment_id, "failed")


@router.post("", response_model=dict)
async def start_adaptation(
    body: AdaptRequest,
    background_tasks: BackgroundTasks,
    task_repo: TaskRepository = Depends(get_task_repo),
    exp_repo: ExperimentRepository = Depends(get_experiment_repo),
) -> dict:
    task = await task_repo.get_by_id(body.task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    # Verify model exists via the session already open in exp_repo
    session = exp_repo._session
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

    # Pass engine reference so the background task can create its own sessions
    engine = session.get_bind()
    background_tasks.add_task(_run_adaptation, engine, experiment.experiment_id)

    return {"job_id": str(experiment.experiment_id)}
