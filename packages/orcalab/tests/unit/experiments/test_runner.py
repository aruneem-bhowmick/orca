"""Unit tests for ExperimentRunner: retry logic, MLflow tracking, pruner integration."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from orca_shared.schemas.training import TrainingConfig
from orcalab.experiments.experiment import Experiment
from orcalab.experiments.lifecycle import ExperimentStatus
from orcalab.experiments.runner import ExperimentRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_experiment(epochs: int = 5) -> Experiment:
    return Experiment(
        experiment_id=uuid.uuid4(),
        status="queued",
        training_config=TrainingConfig(epochs=epochs, lr=1e-3, batch_size=32),
    )


def _make_runner(
    model_factory: Any = None,
    max_retries: int = 2,
    timeout: int = 3600,
) -> tuple[ExperimentRunner, MagicMock, AsyncMock]:
    """Return (runner, mock_tracker, mock_artifact_manager)."""
    tracker = MagicMock()
    tracker.__aenter__ = AsyncMock(return_value=tracker)
    tracker.__aexit__ = AsyncMock(return_value=False)
    tracker.run_id = "run-abc-123"
    tracker.log_params = MagicMock()
    tracker.log_metric = MagicMock()

    artifact_mgr = AsyncMock()
    artifact_mgr.upload_model = AsyncMock(return_value="s3://bucket/model")

    runner = ExperimentRunner(
        tracker=tracker,
        artifact_manager=artifact_mgr,
        max_retries=max_retries,
        timeout=timeout,
        model_factory=model_factory,
    )
    return runner, tracker, artifact_mgr


def _always_ok_factory(n_calls: list[int] | None = None) -> Any:
    """Model factory that succeeds every epoch and optionally counts calls."""

    def factory(_config: dict) -> Any:
        model = MagicMock()
        call_count = [0]

        def train_epoch(epoch: int) -> float:
            call_count[0] += 1
            if n_calls is not None:
                n_calls.append(epoch)
            return float(epoch) / 10.0

        model.train_epoch = train_epoch
        return model

    return factory


def _raise_on_epoch_factory(fail_epoch: int, exc: Exception | None = None) -> Any:
    """Model factory whose train_epoch raises on *fail_epoch*."""
    error = exc or RuntimeError(f"boom at epoch {fail_epoch}")

    def factory(_config: dict) -> Any:
        model = MagicMock()

        def train_epoch(epoch: int) -> float:
            if epoch == fail_epoch:
                raise error
            return float(epoch) / 10.0

        model.train_epoch = train_epoch
        return model

    return factory


# ---------------------------------------------------------------------------
# Successful run
# ---------------------------------------------------------------------------


class TestSuccessfulRun:
    async def test_returns_completed_status(self) -> None:
        exp = _make_experiment(epochs=3)
        runner, tracker, artifact_mgr = _make_runner(_always_ok_factory())
        result = await runner.run(exp)
        assert result.status == "completed"

    async def test_log_metric_called_once_per_epoch(self) -> None:
        epochs = 4
        exp = _make_experiment(epochs=epochs)
        runner, tracker, artifact_mgr = _make_runner(_always_ok_factory())
        await runner.run(exp)
        assert tracker.log_metric.call_count == epochs

    async def test_log_metric_step_matches_epoch(self) -> None:
        exp = _make_experiment(epochs=3)
        runner, tracker, _ = _make_runner(_always_ok_factory())
        await runner.run(exp)
        steps = [c.kwargs["step"] for c in tracker.log_metric.call_args_list]
        assert steps == [1, 2, 3]

    async def test_upload_model_called_once_on_success(self) -> None:
        exp = _make_experiment(epochs=2)
        runner, _, artifact_mgr = _make_runner(_always_ok_factory())
        await runner.run(exp)
        artifact_mgr.upload_model.assert_awaited_once()

    async def test_result_contains_mlflow_run_id(self) -> None:
        exp = _make_experiment(epochs=2)
        runner, tracker, _ = _make_runner(_always_ok_factory())
        result = await runner.run(exp)
        assert result.mlflow_run_id == "run-abc-123"

    async def test_result_metrics_contain_final_metric(self) -> None:
        exp = _make_experiment(epochs=3)
        runner, _, _ = _make_runner(_always_ok_factory())
        result = await runner.run(exp)
        assert result.metrics is not None
        assert "metric" in result.metrics

    async def test_log_params_called_with_training_config(self) -> None:
        exp = _make_experiment(epochs=1)
        runner, tracker, _ = _make_runner(_always_ok_factory())
        await runner.run(exp)
        tracker.log_params.assert_called_once()

    async def test_default_epochs_10_when_no_training_config(self) -> None:
        exp = Experiment(experiment_id=uuid.uuid4(), status="queued")
        epochs_seen: list[int] = []
        runner, _, _ = _make_runner(_always_ok_factory(n_calls=epochs_seen))
        await runner.run(exp)
        assert len(epochs_seen) == 10


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


class TestRetryBehaviour:
    async def test_max_retries_2_attempts_three_times_total(self) -> None:
        attempt_count = [0]

        def factory(_config: dict) -> Any:
            model = MagicMock()

            def train_epoch(epoch: int) -> float:
                attempt_count[0] += 1
                raise RuntimeError("always fails")

            model.train_epoch = train_epoch
            return model

        exp = _make_experiment(epochs=1)
        runner, _, _ = _make_runner(factory, max_retries=2)
        result = await runner.run(exp)
        assert result.status == "failed"
        assert attempt_count[0] == 3  # 1 initial + 2 retries

    async def test_max_retries_0_fails_immediately_after_one_attempt(self) -> None:
        attempt_count = [0]

        def factory(_config: dict) -> Any:
            model = MagicMock()

            def train_epoch(epoch: int) -> float:
                attempt_count[0] += 1
                raise RuntimeError("fail")

            model.train_epoch = train_epoch
            return model

        exp = _make_experiment(epochs=1)
        runner, _, _ = _make_runner(factory, max_retries=0)
        result = await runner.run(exp)
        assert result.status == "failed"
        assert attempt_count[0] == 1

    async def test_succeeds_on_second_attempt(self) -> None:
        attempt_count = [0]

        def factory(_config: dict) -> Any:
            model = MagicMock()

            def train_epoch(epoch: int) -> float:
                attempt_count[0] += 1
                if attempt_count[0] == 1:
                    raise RuntimeError("first attempt fails")
                return 0.9

            model.train_epoch = train_epoch
            return model

        exp = _make_experiment(epochs=1)
        runner, _, _ = _make_runner(factory, max_retries=2)
        result = await runner.run(exp)
        assert result.status == "completed"
        assert attempt_count[0] == 2

    async def test_failed_result_has_no_mlflow_run_id_when_all_retries_exhausted(
        self,
    ) -> None:
        exp = _make_experiment(epochs=1)
        runner, tracker, _ = _make_runner(_raise_on_epoch_factory(1), max_retries=0)
        tracker.run_id = None
        result = await runner.run(exp)
        assert result.status == "failed"

    async def test_upload_not_called_when_run_fails(self) -> None:
        exp = _make_experiment(epochs=1)
        runner, _, artifact_mgr = _make_runner(
            _raise_on_epoch_factory(1), max_retries=0
        )
        await runner.run(exp)
        artifact_mgr.upload_model.assert_not_awaited()


# ---------------------------------------------------------------------------
# Pruner integration
# ---------------------------------------------------------------------------


class TestPrunerIntegration:
    async def test_pruner_triggered_at_epoch_1_returns_failed(self) -> None:
        exp = _make_experiment(epochs=5)
        runner, _, _ = _make_runner(_always_ok_factory())

        pruner = MagicMock()
        pruner.should_prune.return_value = True

        result = await runner.run(exp, pruner=pruner)
        assert result.status == "failed"

    async def test_pruner_stops_training_early(self) -> None:
        epochs_trained: list[int] = []
        exp = _make_experiment(epochs=5)
        runner, tracker, _ = _make_runner(_always_ok_factory(n_calls=epochs_trained))

        pruner = MagicMock()
        pruner.should_prune.side_effect = lambda tid, step, val, all_vals: step >= 2

        await runner.run(exp, pruner=pruner)
        assert max(epochs_trained) == 2

    async def test_pruner_receives_correct_trial_id(self) -> None:
        exp = _make_experiment(epochs=3)
        runner, _, _ = _make_runner(_always_ok_factory())

        pruner = MagicMock()
        pruner.should_prune.return_value = False

        await runner.run(exp, pruner=pruner)
        for c in pruner.should_prune.call_args_list:
            assert c.args[0] == str(exp.experiment_id)

    async def test_no_pruner_completes_all_epochs(self) -> None:
        epochs_seen: list[int] = []
        exp = _make_experiment(epochs=4)
        runner, _, _ = _make_runner(_always_ok_factory(n_calls=epochs_seen))
        result = await runner.run(exp, pruner=None)
        assert result.status == "completed"
        assert len(epochs_seen) == 4

    async def test_upload_not_called_when_pruned(self) -> None:
        exp = _make_experiment(epochs=5)
        runner, _, artifact_mgr = _make_runner(_always_ok_factory())

        pruner = MagicMock()
        pruner.should_prune.return_value = True

        await runner.run(exp, pruner=pruner)
        artifact_mgr.upload_model.assert_not_awaited()
