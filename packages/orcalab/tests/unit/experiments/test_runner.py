"""Unit tests for ExperimentRunner: retry logic, MLflow tracking, pruner integration."""

from __future__ import annotations

import asyncio
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

    async def test_result_metrics_contain_loss(self) -> None:
        exp = _make_experiment(epochs=3)
        runner, _, _ = _make_runner(_always_ok_factory())
        result = await runner.run(exp)
        assert result.metrics is not None
        assert "loss" in result.metrics

    async def test_result_metrics_do_not_use_generic_metric_key(self) -> None:
        exp = _make_experiment(epochs=3)
        runner, _, _ = _make_runner(_always_ok_factory())
        result = await runner.run(exp)
        assert result.metrics is not None
        assert "metric" not in result.metrics

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
        assert result.mlflow_run_id is None

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


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestRunnerConstruction:
    def test_negative_max_retries_raises(self) -> None:
        tracker = MagicMock()
        artifact_mgr = AsyncMock()
        with pytest.raises(ValueError, match="max_retries"):
            ExperimentRunner(tracker=tracker, artifact_manager=artifact_mgr, max_retries=-1)

    def test_zero_timeout_raises(self) -> None:
        tracker = MagicMock()
        artifact_mgr = AsyncMock()
        with pytest.raises(ValueError, match="timeout"):
            ExperimentRunner(tracker=tracker, artifact_manager=artifact_mgr, timeout=0)

    def test_negative_timeout_raises(self) -> None:
        tracker = MagicMock()
        artifact_mgr = AsyncMock()
        with pytest.raises(ValueError, match="timeout"):
            ExperimentRunner(tracker=tracker, artifact_manager=artifact_mgr, timeout=-100)

    def test_zero_max_retries_is_valid(self) -> None:
        tracker = MagicMock()
        artifact_mgr = AsyncMock()
        runner = ExperimentRunner(tracker=tracker, artifact_manager=artifact_mgr, max_retries=0)
        assert runner._max_retries == 0

    def test_default_values_are_valid(self) -> None:
        tracker = MagicMock()
        artifact_mgr = AsyncMock()
        runner = ExperimentRunner(tracker=tracker, artifact_manager=artifact_mgr)
        assert runner._max_retries == 2
        assert runner._timeout == 3600


# ---------------------------------------------------------------------------
# Timeout behaviour
# ---------------------------------------------------------------------------


class TestTimeoutBehaviour:
    """Tests covering the asyncio.TimeoutError branch inside ExperimentRunner.run.

    Each test patches asyncio.wait_for at the module level so no real coroutine
    is ever suspended; the mock explicitly closes the passed coroutine before
    raising to avoid ResourceWarning from unclosed awaitables.
    """

    async def test_timeout_returns_failed_status(self) -> None:
        """A single timed-out attempt yields status 'failed'."""
        exp = _make_experiment(epochs=5)
        runner, _, _ = _make_runner(_always_ok_factory())

        async def _close_and_timeout(coro, timeout):
            coro.close()
            raise asyncio.TimeoutError

        with patch(
            "orcalab.experiments.runner.asyncio.wait_for",
            side_effect=_close_and_timeout,
        ):
            result = await runner.run(exp)

        assert result.status == "failed"

    async def test_timeout_retries_and_returns_failed_after_exhaustion(self) -> None:
        """With max_retries=2, wait_for is called exactly 3 times before the runner gives up."""
        exp = _make_experiment(epochs=5)
        runner, _, _ = _make_runner(_always_ok_factory(), max_retries=2)
        wait_for_call_count = [0]

        async def _raise_timeout(coro, *_args, **_kwargs):
            wait_for_call_count[0] += 1
            coro.close()
            raise asyncio.TimeoutError

        with patch("orcalab.experiments.runner.asyncio.wait_for", side_effect=_raise_timeout):
            result = await runner.run(exp)

        assert result.status == "failed"
        # 1 initial attempt + 2 retries = 3 total calls
        assert wait_for_call_count[0] == 3

    async def test_timeout_with_zero_retries_fails_after_one_attempt(self) -> None:
        """max_retries=0 stops immediately after the first timed-out attempt."""
        exp = _make_experiment(epochs=5)
        runner, _, _ = _make_runner(_always_ok_factory(), max_retries=0)
        call_count = [0]

        async def _raise_timeout(coro, *_args, **_kwargs):
            call_count[0] += 1
            coro.close()
            raise asyncio.TimeoutError

        with patch("orcalab.experiments.runner.asyncio.wait_for", side_effect=_raise_timeout):
            result = await runner.run(exp)

        assert result.status == "failed"
        assert call_count[0] == 1

    async def test_timeout_does_not_call_upload_model(self) -> None:
        """Artifact upload must never be attempted when every attempt times out."""
        exp = _make_experiment(epochs=5)
        runner, _, artifact_mgr = _make_runner(_always_ok_factory(), max_retries=0)

        async def _close_and_timeout(coro, timeout):
            coro.close()
            raise asyncio.TimeoutError

        with patch(
            "orcalab.experiments.runner.asyncio.wait_for",
            side_effect=_close_and_timeout,
        ):
            await runner.run(exp)

        artifact_mgr.upload_model.assert_not_awaited()

    async def test_succeeds_after_initial_timeout_if_retry_works(self) -> None:
        """A timeout on attempt 1 followed by a clean attempt 2 yields status 'completed'."""
        exp = _make_experiment(epochs=2)
        runner, tracker, _ = _make_runner(_always_ok_factory(), max_retries=2)
        call_count = [0]

        original_wait_for = asyncio.wait_for

        async def _fail_first_then_succeed(coro, timeout):
            call_count[0] += 1
            if call_count[0] == 1:
                coro.close()
                raise asyncio.TimeoutError
            return await original_wait_for(coro, timeout=timeout)

        with patch("orcalab.experiments.runner.asyncio.wait_for", side_effect=_fail_first_then_succeed):
            result = await runner.run(exp)

        assert result.status == "completed"
        assert call_count[0] == 2


# ---------------------------------------------------------------------------
# Epoch tracking and metric naming
# ---------------------------------------------------------------------------


def _make_runner_with_repo(
    model_factory: Any = None,
    max_retries: int = 0,
    timeout: int = 3600,
) -> tuple["ExperimentRunner", MagicMock, AsyncMock, AsyncMock]:
    """Return (runner, mock_tracker, mock_artifact_manager, mock_repository)."""
    tracker = MagicMock()
    tracker.__aenter__ = AsyncMock(return_value=tracker)
    tracker.__aexit__ = AsyncMock(return_value=False)
    tracker.run_id = "run-epoch-test"
    tracker.log_params = MagicMock()
    tracker.log_metric = MagicMock()

    artifact_mgr = AsyncMock()
    artifact_mgr.upload_model = AsyncMock(return_value="s3://bucket/model")

    repo = AsyncMock()
    repo.update_status = AsyncMock()
    repo.update_status_if_current = AsyncMock(return_value=True)
    repo.update_metrics = AsyncMock()

    runner = ExperimentRunner(
        tracker=tracker,
        artifact_manager=artifact_mgr,
        max_retries=max_retries,
        timeout=timeout,
        model_factory=model_factory or _always_ok_factory(),
        repository=repo,
    )
    return runner, tracker, artifact_mgr, repo


class TestEpochTracking:
    async def test_result_metrics_contain_epoch(self) -> None:
        exp = _make_experiment(epochs=4)
        runner, _, _, _ = _make_runner_with_repo(_always_ok_factory())
        result = await runner.run(exp)
        assert result.metrics is not None
        assert "epoch" in result.metrics

    async def test_epoch_equals_total_epochs_on_success(self) -> None:
        total = 5
        exp = _make_experiment(epochs=total)
        runner, _, _, _ = _make_runner_with_repo(_always_ok_factory())
        result = await runner.run(exp)
        assert result.metrics is not None
        assert result.metrics["epoch"] == total

    async def test_loss_key_in_result_metrics(self) -> None:
        exp = _make_experiment(epochs=3)
        runner, _, _, _ = _make_runner_with_repo(_always_ok_factory())
        result = await runner.run(exp)
        assert result.metrics is not None
        assert "loss" in result.metrics

    async def test_tracker_log_metric_uses_loss_name(self) -> None:
        exp = _make_experiment(epochs=3)
        runner, tracker, _, _ = _make_runner_with_repo(_always_ok_factory())
        await runner.run(exp)
        names_logged = [c.args[0] for c in tracker.log_metric.call_args_list]
        assert all(n == "loss" for n in names_logged)
        assert "metric" not in names_logged

    async def test_repository_update_metrics_called_once_per_epoch(self) -> None:
        epochs = 4
        exp = _make_experiment(epochs=epochs)
        runner, _, _, repo = _make_runner_with_repo(_always_ok_factory())
        await runner.run(exp)
        assert repo.update_metrics.await_count == epochs

    async def test_repository_update_metrics_receives_correct_epoch_numbers(self) -> None:
        exp = _make_experiment(epochs=3)
        runner, _, _, repo = _make_runner_with_repo(_always_ok_factory())
        await runner.run(exp)
        epoch_args = [c.args[1]["epoch"] for c in repo.update_metrics.call_args_list]
        assert epoch_args == [1, 2, 3]

    async def test_repository_update_metrics_receives_loss_key(self) -> None:
        exp = _make_experiment(epochs=2)
        runner, _, _, repo = _make_runner_with_repo(_always_ok_factory())
        await runner.run(exp)
        for c in repo.update_metrics.call_args_list:
            metrics_dict = c.args[1]
            assert "loss" in metrics_dict

    async def test_null_repository_does_not_raise(self) -> None:
        exp = _make_experiment(epochs=3)
        runner, _, _ = _make_runner(_always_ok_factory())  # uses _NullRepository
        result = await runner.run(exp)
        assert result.status == "completed"

    async def test_loss_value_matches_model_train_epoch_return(self) -> None:
        exp = _make_experiment(epochs=3)
        runner, _, _, _ = _make_runner_with_repo(_always_ok_factory())
        result = await runner.run(exp)
        assert result.metrics is not None
        # _always_ok_factory returns epoch/10.0; last epoch is 3 → 0.3
        assert result.metrics["loss"] == pytest.approx(3 / 10.0)
