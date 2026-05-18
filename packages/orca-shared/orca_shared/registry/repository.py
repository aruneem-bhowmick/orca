from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from orca_shared.registry.models import (
    Embedding as EmbeddingORM,
    Experiment as ExperimentORM,
    Model as ModelORM,
    Performance as PerformanceORM,
    SearchSpace as SearchSpaceORM,
    Task as TaskORM,
)
from orca_shared.schemas.embedding import Embedding
from orca_shared.schemas.metrics import MetricPoint, PerformanceMetrics, PerformanceSummary
from orca_shared.schemas.search_space import SearchSpaceRecord
from orca_shared.schemas.task import Task, TaskCreate, TaskSummary
from orca_shared.schemas.training import ExperimentResult


class TaskRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, data: TaskCreate) -> Task:
        row = TaskORM(
            task_id=uuid.uuid4(),
            name=data.name,
            domain=data.domain,
            task_type=data.task_type,
            n_samples=data.n_samples,
            n_features=data.n_features,
            n_classes=data.n_classes,
            dataset_uri=data.dataset_uri,
            task_metadata=data.metadata,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self._session.add(row)
        await self._session.flush()
        return Task.model_validate(row)

    async def get_by_id(self, task_id: uuid.UUID) -> Optional[Task]:
        result = await self._session.execute(
            select(TaskORM).where(TaskORM.task_id == task_id)
        )
        row = result.scalar_one_or_none()
        return Task.model_validate(row) if row is not None else None

    async def list_by_domain(
        self, domain: str, *, limit: int = 500, offset: int = 0
    ) -> list[TaskSummary]:
        result = await self._session.execute(
            select(TaskORM).where(TaskORM.domain == domain).limit(limit).offset(offset)
        )
        return [TaskSummary.model_validate(r) for r in result.scalars()]

    async def list_by_type(
        self, task_type: str, *, limit: int = 500, offset: int = 0
    ) -> list[TaskSummary]:
        result = await self._session.execute(
            select(TaskORM).where(TaskORM.task_type == task_type).limit(limit).offset(offset)
        )
        return [TaskSummary.model_validate(r) for r in result.scalars()]

    async def list_all(
        self, *, limit: int = 500, offset: int = 0
    ) -> list[TaskSummary]:
        result = await self._session.execute(
            select(TaskORM).limit(limit).offset(offset)
        )
        return [TaskSummary.model_validate(r) for r in result.scalars()]

    async def update_embedding(self, task_id: uuid.UUID, embedding_id: uuid.UUID) -> None:
        await self._session.execute(
            update(TaskORM)
            .where(TaskORM.task_id == task_id)
            .values(embedding_id=embedding_id, updated_at=datetime.now(timezone.utc))
        )


class ExperimentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        task_id: uuid.UUID,
        model_id: uuid.UUID,
        training_config: dict[str, Any] | None = None,
        created_by: str | None = None,
    ) -> ExperimentResult:
        row = ExperimentORM(
            experiment_id=uuid.uuid4(),
            task_id=task_id,
            model_id=model_id,
            training_config=training_config,
            status="pending",
            created_by=created_by,
        )
        self._session.add(row)
        await self._session.flush()
        return ExperimentResult.model_validate(row)

    async def get_by_id(self, experiment_id: uuid.UUID) -> Optional[ExperimentResult]:
        result = await self._session.execute(
            select(ExperimentORM).where(ExperimentORM.experiment_id == experiment_id)
        )
        row = result.scalar_one_or_none()
        return ExperimentResult.model_validate(row) if row is not None else None

    async def list_by_task(
        self, task_id: uuid.UUID, *, limit: int = 500, offset: int = 0
    ) -> list[ExperimentResult]:
        result = await self._session.execute(
            select(ExperimentORM)
            .where(ExperimentORM.task_id == task_id)
            .limit(limit)
            .offset(offset)
        )
        return [ExperimentResult.model_validate(r) for r in result.scalars()]

    async def update_status(self, experiment_id: uuid.UUID, status: str) -> None:
        await self._session.execute(
            update(ExperimentORM)
            .where(ExperimentORM.experiment_id == experiment_id)
            .values(status=status)
        )

    async def update_status_if_current(
        self, experiment_id: uuid.UUID, from_status: str, to_status: str
    ) -> bool:
        """Update status only when the current DB status matches *from_status*.

        Returns True when the row was updated, False when another writer already
        changed the status (optimistic concurrency conflict).
        """
        result = await self._session.execute(
            update(ExperimentORM)
            .where(
                ExperimentORM.experiment_id == experiment_id,
                ExperimentORM.status == from_status,
            )
            .values(status=to_status)
        )
        return result.rowcount > 0

    async def list_all(
        self, *, limit: int = 500, offset: int = 0
    ) -> list[ExperimentResult]:
        result = await self._session.execute(
            select(ExperimentORM)
            .order_by(ExperimentORM.experiment_id)
            .limit(limit)
            .offset(offset)
        )
        return [ExperimentResult.model_validate(r) for r in result.scalars()]

    async def update_metrics(
        self, experiment_id: uuid.UUID, metrics: dict[str, Any]
    ) -> None:
        """Merge *metrics* into the stored experiment metrics dict.

        Merges rather than replaces so per-epoch writes accumulate without
        losing keys written by earlier epochs.
        """
        result = await self._session.execute(
            select(ExperimentORM)
            .where(ExperimentORM.experiment_id == experiment_id)
            .with_for_update()
        )
        row = result.scalar_one_or_none()
        if row is not None:
            existing: dict[str, Any] = dict(row.metrics or {})
            existing.update(metrics)
            row.metrics = existing
            await self._session.flush()

    async def mark_complete(
        self, experiment_id: uuid.UUID, mlflow_run_id: str
    ) -> None:
        await self._session.execute(
            update(ExperimentORM)
            .where(ExperimentORM.experiment_id == experiment_id)
            .values(
                status="completed",
                mlflow_run_id=mlflow_run_id,
                completed_at=datetime.now(timezone.utc),
            )
        )


class PerformanceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def log_metric(
        self,
        experiment_id: uuid.UUID,
        name: str,
        value: float,
        epoch: int | None = None,
        is_final: bool = False,
    ) -> MetricPoint:
        row = PerformanceORM(
            performance_id=uuid.uuid4(),
            experiment_id=experiment_id,
            metric_name=name,
            metric_value=value,
            epoch=epoch,
            is_final=is_final,
        )
        self._session.add(row)
        await self._session.flush()
        return MetricPoint(name=name, value=value, step=epoch, is_final=is_final)

    async def get_final_metrics(self, experiment_id: uuid.UUID) -> PerformanceMetrics:
        result = await self._session.execute(
            select(PerformanceORM).where(
                PerformanceORM.experiment_id == experiment_id,
                PerformanceORM.is_final.is_(True),
            )
        )
        rows = list(result.scalars())
        final_metrics = {
            r.metric_name: r.metric_value for r in rows if r.metric_name is not None
        }
        best_epoch = max((r.epoch for r in rows if r.epoch is not None), default=None)
        return PerformanceMetrics(
            experiment_id=experiment_id,
            final_metrics=final_metrics,
            best_epoch=best_epoch,
        )

    async def get_history(
        self, experiment_id: uuid.UUID, metric_name: str
    ) -> list[MetricPoint]:
        result = await self._session.execute(
            select(PerformanceORM).where(
                PerformanceORM.experiment_id == experiment_id,
                PerformanceORM.metric_name == metric_name,
            )
        )
        return [
            MetricPoint(
                name=r.metric_name,
                value=r.metric_value if r.metric_value is not None else 0.0,
                step=r.epoch,
                is_final=r.is_final,
            )
            for r in result.scalars()
        ]

    async def list_all_with_context(
        self, metric_name: str = "accuracy"
    ) -> list[PerformanceSummary]:
        result = await self._session.execute(
            select(
                TaskORM.name.label("task_name"),
                ModelORM.architecture.label("architecture"),
                func.avg(PerformanceORM.metric_value).label("mean_accuracy"),
            )
            .join(ExperimentORM, PerformanceORM.experiment_id == ExperimentORM.experiment_id)
            .join(TaskORM, ExperimentORM.task_id == TaskORM.task_id)
            .outerjoin(ModelORM, ExperimentORM.model_id == ModelORM.model_id)
            .where(PerformanceORM.metric_name == metric_name)
            .group_by(TaskORM.name, ModelORM.architecture)
            .order_by(TaskORM.name, ModelORM.architecture)
        )
        return [
            PerformanceSummary(
                task_name=row.task_name,
                architecture=row.architecture or "unknown",
                mean_accuracy=float(row.mean_accuracy) if row.mean_accuracy is not None else 0.0,
            )
            for row in result.all()
        ]


class SearchSpaceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        name: str | None,
        definition: dict,
    ) -> SearchSpaceRecord:
        row = SearchSpaceORM(
            search_space_id=uuid.uuid4(),
            name=name,
            definition=definition,
            created_at=datetime.now(timezone.utc),
        )
        self._session.add(row)
        await self._session.flush()
        return SearchSpaceRecord.model_validate(row)

    async def list_all(
        self, *, limit: int = 500, offset: int = 0
    ) -> list[SearchSpaceRecord]:
        result = await self._session.execute(
            select(SearchSpaceORM)
            .order_by(SearchSpaceORM.search_space_id)
            .limit(limit)
            .offset(offset)
        )
        return [SearchSpaceRecord.model_validate(r) for r in result.scalars()]


class EmbeddingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        task_id: uuid.UUID,
        embedding_vector: list[float],
        embedding_type: str = "statistical",
        model_version: str = "v1",
    ) -> Embedding:
        row = EmbeddingORM(
            embedding_id=uuid.uuid4(),
            task_id=task_id,
            embedding_type=embedding_type,
            embedding_vector=embedding_vector,
            dimension=len(embedding_vector),
            model_version=model_version,
            created_at=datetime.now(timezone.utc),
        )
        self._session.add(row)
        await self._session.flush()
        return Embedding.model_validate(row)

    async def get_by_id(self, embedding_id: uuid.UUID) -> Optional[Embedding]:
        result = await self._session.execute(
            select(EmbeddingORM).where(EmbeddingORM.embedding_id == embedding_id)
        )
        row = result.scalar_one_or_none()
        return Embedding.model_validate(row) if row is not None else None
