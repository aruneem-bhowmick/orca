"""Tests for TaskRepository, ExperimentRepository, and PerformanceRepository.

All tests use a mocked AsyncSession — no database connection required.
The conftest helpers make_task_row / make_experiment_row / make_performance_row
produce SimpleNamespace objects that satisfy model_validate(from_attributes=True).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from tests.conftest import (
    make_embedding_row,
    make_execute_result,
    make_experiment_row,
    make_performance_row,
    make_task_row,
)
from orca_shared.registry.repository import (
    EmbeddingRepository,
    ExperimentRepository,
    PerformanceRepository,
    TaskRepository,
)
from orca_shared.schemas.embedding import Embedding
from orca_shared.schemas.metrics import MetricPoint, PerformanceMetrics
from orca_shared.schemas.task import Task, TaskCreate, TaskSummary
from orca_shared.schemas.training import ExperimentResult


# ---------------------------------------------------------------------------
# TaskRepository
# ---------------------------------------------------------------------------


class TestTaskRepositoryCreate:
    async def test_add_and_flush_called(self, mock_session):
        data = TaskCreate(name="iris", task_type="classification")
        await TaskRepository(mock_session).create(data)
        mock_session.add.assert_called_once()
        mock_session.flush.assert_awaited_once()

    async def test_returns_task_schema(self, mock_session):
        data = TaskCreate(name="iris", task_type="classification", domain="tabular")
        result = await TaskRepository(mock_session).create(data)
        assert isinstance(result, Task)
        assert result.name == "iris"
        assert result.task_type == "classification"
        assert result.domain == "tabular"

    async def test_task_id_is_uuid(self, mock_session):
        data = TaskCreate(name="wine", task_type="classification")
        result = await TaskRepository(mock_session).create(data)
        assert isinstance(result.task_id, uuid.UUID)

    async def test_metadata_field_correctly_mapped(self, mock_session):
        """Regression: task_metadata ORM attr must flow into schema metadata field."""
        data = TaskCreate(name="t", task_type="clf", metadata={"source": "openml"})
        result = await TaskRepository(mock_session).create(data)
        assert result.metadata == {"source": "openml"}

    async def test_optional_fields_default_none(self, mock_session):
        data = TaskCreate(name="bare", task_type="regression")
        result = await TaskRepository(mock_session).create(data)
        assert result.domain is None
        assert result.n_samples is None
        assert result.n_features is None

    async def test_created_at_set_to_utc_now(self, mock_session):
        data = TaskCreate(name="t", task_type="clf")
        before = datetime.now(timezone.utc)
        result = await TaskRepository(mock_session).create(data)
        after = datetime.now(timezone.utc)
        assert before <= result.created_at <= after


class TestTaskRepositoryGetById:
    async def test_returns_task_when_found(self, mock_session, task_id):
        row = make_task_row(task_id=task_id, name="iris", task_type="classification")
        mock_session.execute.return_value = make_execute_result([row])

        result = await TaskRepository(mock_session).get_by_id(task_id)

        assert result is not None
        assert isinstance(result, Task)
        assert result.task_id == task_id
        assert result.name == "iris"

    async def test_returns_none_when_not_found(self, mock_session):
        mock_session.execute.return_value = make_execute_result([])

        result = await TaskRepository(mock_session).get_by_id(uuid.uuid4())

        assert result is None

    async def test_orm_metadata_alias_resolved(self, mock_session, task_id):
        """task_metadata on the ORM row must appear as metadata in the schema."""
        row = make_task_row(task_id=task_id, task_metadata={"flag": True})
        mock_session.execute.return_value = make_execute_result([row])

        result = await TaskRepository(mock_session).get_by_id(task_id)

        assert result.metadata == {"flag": True}

    async def test_execute_called_once(self, mock_session, task_id):
        mock_session.execute.return_value = make_execute_result([])
        await TaskRepository(mock_session).get_by_id(task_id)
        mock_session.execute.assert_awaited_once()


class TestTaskRepositoryListByDomain:
    async def test_returns_task_summaries(self, mock_session):
        rows = [
            make_task_row(name="iris", domain="tabular", task_type="classification"),
            make_task_row(name="wine", domain="tabular", task_type="classification"),
        ]
        mock_session.execute.return_value = make_execute_result(rows)

        results = await TaskRepository(mock_session).list_by_domain("tabular")

        assert len(results) == 2
        assert all(isinstance(r, TaskSummary) for r in results)
        names = {r.name for r in results}
        assert names == {"iris", "wine"}

    async def test_empty_domain_returns_empty_list(self, mock_session):
        mock_session.execute.return_value = make_execute_result([])
        results = await TaskRepository(mock_session).list_by_domain("nonexistent")
        assert results == []


class TestTaskRepositoryListByType:
    async def test_returns_task_summaries_for_type(self, mock_session):
        rows = [make_task_row(task_type="regression")]
        mock_session.execute.return_value = make_execute_result(rows)

        results = await TaskRepository(mock_session).list_by_type("regression")

        assert len(results) == 1
        assert results[0].task_type == "regression"


class TestTaskRepositoryUpdateEmbedding:
    async def test_calls_execute(self, mock_session):
        tid = uuid.uuid4()
        eid = uuid.uuid4()
        mock_session.execute.return_value = MagicMock()

        await TaskRepository(mock_session).update_embedding(tid, eid)

        mock_session.execute.assert_awaited_once()

    async def test_returns_none(self, mock_session):
        mock_session.execute.return_value = MagicMock()
        result = await TaskRepository(mock_session).update_embedding(uuid.uuid4(), uuid.uuid4())
        assert result is None


# ---------------------------------------------------------------------------
# ExperimentRepository
# ---------------------------------------------------------------------------


class TestExperimentRepositoryCreate:
    async def test_add_and_flush_called(self, mock_session, task_id, model_id):
        await ExperimentRepository(mock_session).create(task_id, model_id)
        mock_session.add.assert_called_once()
        mock_session.flush.assert_awaited_once()

    async def test_initial_status_is_pending(self, mock_session, task_id, model_id):
        """New experiments must start in 'pending' status."""
        repo = ExperimentRepository(mock_session)
        result = await repo.create(task_id, model_id)
        assert result.status == "pending"

    async def test_returns_experiment_result(self, mock_session, task_id, model_id):
        result = await ExperimentRepository(mock_session).create(task_id, model_id)
        assert isinstance(result, ExperimentResult)
        assert result.task_id == task_id
        assert result.model_id == model_id

    async def test_training_config_stored(self, mock_session, task_id, model_id):
        cfg = {"lr": 0.001, "epochs": 10}
        result = await ExperimentRepository(mock_session).create(
            task_id, model_id, training_config=cfg
        )
        # training_config is stored on the ORM row; ExperimentResult has no field for it
        added_row = mock_session.add.call_args.args[0]
        assert added_row.training_config == cfg

    async def test_created_by_stored(self, mock_session, task_id, model_id):
        await ExperimentRepository(mock_session).create(
            task_id, model_id, created_by="alice"
        )
        added_row = mock_session.add.call_args.args[0]
        assert added_row.created_by == "alice"

    async def test_experiment_id_is_uuid(self, mock_session, task_id, model_id):
        result = await ExperimentRepository(mock_session).create(task_id, model_id)
        assert isinstance(result.experiment_id, uuid.UUID)


class TestExperimentRepositoryGetById:
    async def test_returns_experiment_when_found(self, mock_session, experiment_id):
        row = make_experiment_row(experiment_id=experiment_id, status="running")
        mock_session.execute.return_value = make_execute_result([row])

        result = await ExperimentRepository(mock_session).get_by_id(experiment_id)

        assert result is not None
        assert isinstance(result, ExperimentResult)
        assert result.experiment_id == experiment_id
        assert result.status == "running"

    async def test_returns_none_when_not_found(self, mock_session):
        mock_session.execute.return_value = make_execute_result([])
        result = await ExperimentRepository(mock_session).get_by_id(uuid.uuid4())
        assert result is None


class TestExperimentRepositoryListByTask:
    async def test_returns_list(self, mock_session, task_id):
        rows = [
            make_experiment_row(task_id=task_id, status="completed"),
            make_experiment_row(task_id=task_id, status="failed"),
        ]
        mock_session.execute.return_value = make_execute_result(rows)

        results = await ExperimentRepository(mock_session).list_by_task(task_id)

        assert len(results) == 2
        statuses = {r.status for r in results}
        assert statuses == {"completed", "failed"}

    async def test_empty_list(self, mock_session, task_id):
        mock_session.execute.return_value = make_execute_result([])
        results = await ExperimentRepository(mock_session).list_by_task(task_id)
        assert results == []


class TestExperimentRepositoryUpdateStatus:
    async def test_calls_execute(self, mock_session, experiment_id):
        mock_session.execute.return_value = MagicMock()
        await ExperimentRepository(mock_session).update_status(experiment_id, "running")
        mock_session.execute.assert_awaited_once()

    async def test_returns_none(self, mock_session, experiment_id):
        mock_session.execute.return_value = MagicMock()
        result = await ExperimentRepository(mock_session).update_status(experiment_id, "running")
        assert result is None


class TestExperimentRepositoryMarkComplete:
    async def test_calls_execute(self, mock_session, experiment_id):
        mock_session.execute.return_value = MagicMock()
        await ExperimentRepository(mock_session).mark_complete(experiment_id, "mlrun-abc")
        mock_session.execute.assert_awaited_once()

    async def test_returns_none(self, mock_session, experiment_id):
        mock_session.execute.return_value = MagicMock()
        result = await ExperimentRepository(mock_session).mark_complete(
            experiment_id, "mlrun-abc"
        )
        assert result is None


class TestExperimentRepositoryUpdateMetrics:
    async def test_merges_new_keys_into_empty_metrics(self, mock_session, experiment_id):
        row = make_experiment_row(experiment_id=experiment_id, metrics=None)
        mock_session.execute.return_value = make_execute_result([row])

        await ExperimentRepository(mock_session).update_metrics(
            experiment_id, {"loss": 0.5, "epoch": 1}
        )

        assert row.metrics == {"loss": 0.5, "epoch": 1}

    async def test_merges_into_existing_metrics(self, mock_session, experiment_id):
        row = make_experiment_row(experiment_id=experiment_id, metrics={"loss": 0.9, "epoch": 1})
        mock_session.execute.return_value = make_execute_result([row])

        await ExperimentRepository(mock_session).update_metrics(
            experiment_id, {"loss": 0.7, "epoch": 2}
        )

        assert row.metrics["loss"] == pytest.approx(0.7)
        assert row.metrics["epoch"] == 2

    async def test_overwrites_existing_key(self, mock_session, experiment_id):
        row = make_experiment_row(experiment_id=experiment_id, metrics={"loss": 1.0})
        mock_session.execute.return_value = make_execute_result([row])

        await ExperimentRepository(mock_session).update_metrics(experiment_id, {"loss": 0.3})

        assert row.metrics["loss"] == pytest.approx(0.3)

    async def test_preserves_unrelated_keys(self, mock_session, experiment_id):
        row = make_experiment_row(
            experiment_id=experiment_id, metrics={"accuracy": 0.88, "epoch": 5}
        )
        mock_session.execute.return_value = make_execute_result([row])

        await ExperimentRepository(mock_session).update_metrics(
            experiment_id, {"loss": 0.4, "epoch": 6}
        )

        assert row.metrics["accuracy"] == pytest.approx(0.88)
        assert row.metrics["loss"] == pytest.approx(0.4)
        assert row.metrics["epoch"] == 6

    async def test_noop_when_experiment_not_found(self, mock_session):
        mock_session.execute.return_value = make_execute_result([])

        await ExperimentRepository(mock_session).update_metrics(
            uuid.uuid4(), {"loss": 0.5}
        )  # must not raise

    async def test_calls_flush_after_update(self, mock_session, experiment_id):
        row = make_experiment_row(experiment_id=experiment_id, metrics=None)
        mock_session.execute.return_value = make_execute_result([row])

        await ExperimentRepository(mock_session).update_metrics(
            experiment_id, {"loss": 0.2, "epoch": 3}
        )

        mock_session.flush.assert_awaited_once()

    async def test_returns_none(self, mock_session, experiment_id):
        row = make_experiment_row(experiment_id=experiment_id)
        mock_session.execute.return_value = make_execute_result([row])

        result = await ExperimentRepository(mock_session).update_metrics(
            experiment_id, {"loss": 0.1}
        )
        assert result is None


# ---------------------------------------------------------------------------
# PerformanceRepository
# ---------------------------------------------------------------------------


class TestPerformanceRepositoryLogMetric:
    async def test_add_and_flush_called(self, mock_session, experiment_id):
        await PerformanceRepository(mock_session).log_metric(experiment_id, "acc", 0.9)
        mock_session.add.assert_called_once()
        mock_session.flush.assert_awaited_once()

    async def test_returns_metric_point(self, mock_session, experiment_id):
        result = await PerformanceRepository(mock_session).log_metric(
            experiment_id, "accuracy", 0.95
        )
        assert isinstance(result, MetricPoint)
        assert result.name == "accuracy"
        assert result.value == pytest.approx(0.95)

    async def test_epoch_mapped_to_step(self, mock_session, experiment_id):
        result = await PerformanceRepository(mock_session).log_metric(
            experiment_id, "loss", 0.3, epoch=7
        )
        assert result.step == 7

    async def test_is_final_propagated(self, mock_session, experiment_id):
        result = await PerformanceRepository(mock_session).log_metric(
            experiment_id, "f1", 0.88, is_final=True
        )
        assert result.is_final is True

    async def test_is_final_defaults_false(self, mock_session, experiment_id):
        result = await PerformanceRepository(mock_session).log_metric(
            experiment_id, "loss", 0.5
        )
        assert result.is_final is False

    async def test_orm_row_attributes_set(self, mock_session, experiment_id):
        await PerformanceRepository(mock_session).log_metric(
            experiment_id, "acc", 0.9, epoch=3, is_final=True
        )
        row = mock_session.add.call_args.args[0]
        assert row.metric_name == "acc"
        assert row.metric_value == pytest.approx(0.9)
        assert row.epoch == 3
        assert row.is_final is True
        assert row.experiment_id == experiment_id


class TestPerformanceRepositoryGetFinalMetrics:
    async def test_aggregates_final_rows(self, mock_session, experiment_id):
        rows = [
            make_performance_row(
                experiment_id=experiment_id,
                metric_name="accuracy",
                metric_value=0.95,
                epoch=10,
                is_final=True,
            ),
            make_performance_row(
                experiment_id=experiment_id,
                metric_name="f1",
                metric_value=0.88,
                epoch=10,
                is_final=True,
            ),
        ]
        mock_session.execute.return_value = make_execute_result(rows)

        result = await PerformanceRepository(mock_session).get_final_metrics(experiment_id)

        assert isinstance(result, PerformanceMetrics)
        assert result.experiment_id == experiment_id
        assert result.final_metrics["accuracy"] == pytest.approx(0.95)
        assert result.final_metrics["f1"] == pytest.approx(0.88)

    async def test_best_epoch_is_max(self, mock_session, experiment_id):
        rows = [
            make_performance_row(metric_name="acc", metric_value=0.9, epoch=5, is_final=True),
            make_performance_row(metric_name="f1", metric_value=0.85, epoch=12, is_final=True),
        ]
        mock_session.execute.return_value = make_execute_result(rows)

        result = await PerformanceRepository(mock_session).get_final_metrics(experiment_id)

        assert result.best_epoch == 12
        assert set(result.final_metrics) == {"acc", "f1"}

    async def test_empty_result_gives_empty_dict(self, mock_session, experiment_id):
        mock_session.execute.return_value = make_execute_result([])

        result = await PerformanceRepository(mock_session).get_final_metrics(experiment_id)

        assert result.final_metrics == {}
        assert result.best_epoch is None

    async def test_ignores_rows_with_none_metric_name(self, mock_session, experiment_id):
        rows = [
            make_performance_row(metric_name=None, metric_value=0.5, is_final=True),
            make_performance_row(metric_name="acc", metric_value=0.9, epoch=3, is_final=True),
        ]
        mock_session.execute.return_value = make_execute_result(rows)

        result = await PerformanceRepository(mock_session).get_final_metrics(experiment_id)

        assert "acc" in result.final_metrics
        assert None not in result.final_metrics
        assert result.best_epoch == 3

    async def test_best_epoch_none_when_all_epochs_none(self, mock_session, experiment_id):
        rows = [
            make_performance_row(metric_name="acc", metric_value=0.9, epoch=None, is_final=True),
        ]
        mock_session.execute.return_value = make_execute_result(rows)

        result = await PerformanceRepository(mock_session).get_final_metrics(experiment_id)

        assert result.best_epoch is None


class TestPerformanceRepositoryGetHistory:
    async def test_returns_metric_points(self, mock_session, experiment_id):
        rows = [
            make_performance_row(metric_name="loss", metric_value=0.5, epoch=1),
            make_performance_row(metric_name="loss", metric_value=0.3, epoch=2),
            make_performance_row(metric_name="loss", metric_value=0.1, epoch=3),
        ]
        mock_session.execute.return_value = make_execute_result(rows)

        results = await PerformanceRepository(mock_session).get_history(experiment_id, "loss")

        assert len(results) == 3
        assert all(isinstance(r, MetricPoint) for r in results)
        values = [r.value for r in results]
        assert pytest.approx(values) == [0.5, 0.3, 0.1]

    async def test_null_metric_value_defaults_to_zero(self, mock_session, experiment_id):
        rows = [
            make_performance_row(metric_name="loss", metric_value=None, epoch=1),
        ]
        mock_session.execute.return_value = make_execute_result(rows)

        results = await PerformanceRepository(mock_session).get_history(experiment_id, "loss")

        assert results[0].value == pytest.approx(0.0)

    async def test_empty_history(self, mock_session, experiment_id):
        mock_session.execute.return_value = make_execute_result([])
        results = await PerformanceRepository(mock_session).get_history(experiment_id, "acc")
        assert results == []


# ---------------------------------------------------------------------------
# EmbeddingRepository
# ---------------------------------------------------------------------------


class TestEmbeddingRepositoryCreate:
    async def test_add_and_flush_called(self, mock_session, task_id) -> None:
        await EmbeddingRepository(mock_session).create(task_id, [0.1] * 25)
        mock_session.add.assert_called_once()
        mock_session.flush.assert_awaited_once()

    async def test_returns_embedding_schema(self, mock_session, task_id) -> None:
        result = await EmbeddingRepository(mock_session).create(task_id, [0.1] * 25)
        assert isinstance(result, Embedding)

    async def test_embedding_id_is_uuid(self, mock_session, task_id) -> None:
        result = await EmbeddingRepository(mock_session).create(task_id, [0.0] * 25)
        assert isinstance(result.embedding_id, uuid.UUID)

    async def test_dimension_equals_vector_length(self, mock_session, task_id) -> None:
        vec = [0.5] * 25
        result = await EmbeddingRepository(mock_session).create(task_id, vec)
        assert result.dimension == 25

    async def test_embedding_type_stored(self, mock_session, task_id) -> None:
        await EmbeddingRepository(mock_session).create(
            task_id, [0.0] * 25, embedding_type="statistical"
        )
        row = mock_session.add.call_args.args[0]
        assert row.embedding_type == "statistical"

    async def test_model_version_stored(self, mock_session, task_id) -> None:
        await EmbeddingRepository(mock_session).create(
            task_id, [0.0] * 25, model_version="v1"
        )
        row = mock_session.add.call_args.args[0]
        assert row.model_version == "v1"

    async def test_task_id_stored(self, mock_session, task_id) -> None:
        await EmbeddingRepository(mock_session).create(task_id, [0.0] * 25)
        row = mock_session.add.call_args.args[0]
        assert row.task_id == task_id

    async def test_vector_stored_on_row(self, mock_session, task_id) -> None:
        vec = [float(i) for i in range(25)]
        await EmbeddingRepository(mock_session).create(task_id, vec)
        row = mock_session.add.call_args.args[0]
        assert row.embedding_vector == vec

    async def test_dimension_matches_vector_on_row(self, mock_session, task_id) -> None:
        vec = [0.0] * 25
        await EmbeddingRepository(mock_session).create(task_id, vec)
        row = mock_session.add.call_args.args[0]
        assert row.dimension == len(vec)


class TestEmbeddingRepositoryGetById:
    async def test_returns_embedding_when_found(
        self, mock_session, embedding_id, task_id
    ) -> None:
        row = make_embedding_row(embedding_id=embedding_id, task_id=task_id)
        mock_session.execute.return_value = make_execute_result([row])

        result = await EmbeddingRepository(mock_session).get_by_id(embedding_id)

        assert result is not None
        assert isinstance(result, Embedding)
        assert result.embedding_id == embedding_id

    async def test_returns_none_when_not_found(self, mock_session) -> None:
        mock_session.execute.return_value = make_execute_result([])
        result = await EmbeddingRepository(mock_session).get_by_id(uuid.uuid4())
        assert result is None

    async def test_execute_called_once(self, mock_session, embedding_id) -> None:
        mock_session.execute.return_value = make_execute_result([])
        await EmbeddingRepository(mock_session).get_by_id(embedding_id)
        mock_session.execute.assert_awaited_once()
