"""Unit tests for Prefect orchestration task functions."""

from __future__ import annotations

import io
import uuid
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from orca_shared.schemas.training import ExperimentResult, TrainingConfig
from orcalab.experiments.experiment import Experiment
from orcalab.orchestration.tasks.evaluate import evaluate
from orcalab.orchestration.tasks.log_results import log_results
from orcalab.orchestration.tasks.prepare_data import prepare_data
from orcalab.orchestration.tasks.train_model import train_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _make_result(metrics: dict | None = None) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=uuid.uuid4(),
        status="completed",
        metrics=metrics or {"accuracy": 0.9, "loss": 0.1},
    )


def _make_experiment() -> Experiment:
    return Experiment(
        experiment_id=uuid.uuid4(),
        status="pending",
        training_config=TrainingConfig(epochs=2, lr=1e-3, batch_size=8),
    )


def _make_storage(df: pd.DataFrame | None = None) -> AsyncMock:
    df = df if df is not None else pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    storage = AsyncMock()
    storage.download = AsyncMock(return_value=_make_parquet_bytes(df))
    return storage


def _make_runner(result: ExperimentResult | None = None) -> AsyncMock:
    runner = AsyncMock()
    runner.run = AsyncMock(return_value=result or _make_result())
    return runner


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------


class TestPrepareData:
    async def test_returns_dataframe(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        storage = _make_storage(df)
        result = await prepare_data.fn("task-123", storage)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]

    async def test_parquet_roundtrip_preserves_values(self) -> None:
        df = pd.DataFrame({"x": [10, 20, 30]})
        storage = _make_storage(df)
        result = await prepare_data.fn("task-abc", storage)
        assert list(result["x"]) == [10, 20, 30]

    async def test_downloads_with_correct_key(self) -> None:
        storage = _make_storage()
        await prepare_data.fn("my-task-id", storage)
        storage.download.assert_awaited_once_with("datasets/my-task-id/data.parquet")

    async def test_has_retry_decorator(self) -> None:
        assert prepare_data.retries == 2  # type: ignore[attr-defined]

    async def test_has_retry_delay(self) -> None:
        assert prepare_data.retry_delay_seconds == 30  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------


class TestTrainModel:
    async def test_returns_runner_result(self) -> None:
        expected = _make_result({"accuracy": 0.95})
        runner = _make_runner(expected)
        exp = _make_experiment()
        result = await train_model.fn(exp, pruner=None, runner=runner)
        assert result is expected

    async def test_delegates_to_runner_run(self) -> None:
        runner = _make_runner()
        exp = _make_experiment()
        await train_model.fn(exp, pruner=None, runner=runner)
        runner.run.assert_awaited_once_with(exp, pruner=None)

    async def test_passes_pruner_to_runner(self) -> None:
        runner = _make_runner()
        exp = _make_experiment()
        pruner = MagicMock()
        await train_model.fn(exp, pruner=pruner, runner=runner)
        runner.run.assert_awaited_once_with(exp, pruner=pruner)

    async def test_has_timeout_decorator(self) -> None:
        assert train_model.timeout_seconds == 3600  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    async def test_extracts_known_metrics(self) -> None:
        result = _make_result({"accuracy": 0.9, "loss": 0.1})
        metrics = await evaluate.fn(result, metrics=["accuracy", "loss"])
        assert metrics["accuracy"] == pytest.approx(0.9)
        assert metrics["loss"] == pytest.approx(0.1)

    async def test_missing_metric_returns_none(self) -> None:
        result = _make_result({"accuracy": 0.8})
        metrics = await evaluate.fn(result, metrics=["accuracy", "loss"])
        assert metrics["accuracy"] == pytest.approx(0.8)
        assert metrics["loss"] is None

    async def test_default_metrics_list(self) -> None:
        result = _make_result({"accuracy": 0.7, "loss": 0.3})
        metrics = await evaluate.fn(result)
        assert set(metrics.keys()) == {"accuracy", "loss"}

    async def test_empty_result_metrics_returns_nones(self) -> None:
        result = ExperimentResult(experiment_id=uuid.uuid4(), status="completed", metrics=None)
        out = await evaluate.fn(result, metrics=["accuracy"])
        assert out["accuracy"] is None

    async def test_custom_metric_names(self) -> None:
        result = _make_result({"f1": 0.88, "precision": 0.92})
        out = await evaluate.fn(result, metrics=["f1", "precision"])
        assert out["f1"] == pytest.approx(0.88)
        assert out["precision"] == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# log_results
# ---------------------------------------------------------------------------


class TestLogResults:
    async def test_calls_submit_feedback(self) -> None:
        result = _make_result()
        client = AsyncMock()
        client.submit_feedback = AsyncMock(return_value=None)
        await log_results.fn(result, client)
        client.submit_feedback.assert_awaited_once()

    async def test_http_status_error_is_swallowed(self) -> None:
        import httpx
        result = _make_result()
        client = AsyncMock()
        client.submit_feedback = AsyncMock(
            side_effect=httpx.HTTPStatusError("503", request=AsyncMock(), response=AsyncMock())
        )
        await log_results.fn(result, client)

    async def test_connect_error_is_swallowed(self) -> None:
        import httpx
        result = _make_result()
        client = AsyncMock()
        client.submit_feedback = AsyncMock(side_effect=httpx.ConnectError("refused"))
        await log_results.fn(result, client)

    async def test_timeout_error_is_swallowed(self) -> None:
        import httpx
        result = _make_result()
        client = AsyncMock()
        client.submit_feedback = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        await log_results.fn(result, client)

    async def test_feedback_uses_experiment_id(self) -> None:
        result = _make_result({"accuracy": 0.5})
        client = AsyncMock()
        client.submit_feedback = AsyncMock(return_value=None)
        await log_results.fn(result, client)
        req = client.submit_feedback.call_args[0][0]
        assert req.experiment_id == result.experiment_id

    async def test_feedback_metric_is_max_of_metrics(self) -> None:
        result = _make_result({"accuracy": 0.9, "f1": 0.85})
        client = AsyncMock()
        client.submit_feedback = AsyncMock(return_value=None)
        await log_results.fn(result, client)
        req = client.submit_feedback.call_args[0][0]
        assert req.actual_metric == pytest.approx(0.9)
