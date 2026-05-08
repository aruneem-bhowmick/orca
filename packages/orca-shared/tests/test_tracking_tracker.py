"""Tests for OrcaTracker (async context manager) and MetricLogger.

mlflow is injected via sys.modules so deferred 'import mlflow' inside
each method resolves to the mock — no MLflow server needed.
"""
from __future__ import annotations

import pytest

from orca_shared.tracking.logger import MetricLogger
from orca_shared.tracking.tracker import OrcaTracker


# ---------------------------------------------------------------------------
# OrcaTracker
# ---------------------------------------------------------------------------


class TestOrcaTrackerAenter:
    @pytest.mark.asyncio
    async def test_sets_experiment_name(self, mock_mlflow):
        async with OrcaTracker("my-experiment"):
            mock_mlflow.set_experiment.assert_called_once_with("my-experiment")

    @pytest.mark.asyncio
    async def test_starts_run_without_name(self, mock_mlflow):
        async with OrcaTracker("exp"):
            mock_mlflow.start_run.assert_called_once_with(run_name=None)

    @pytest.mark.asyncio
    async def test_starts_run_with_name(self, mock_mlflow):
        async with OrcaTracker("exp", run_name="sweep-001"):
            mock_mlflow.start_run.assert_called_once_with(run_name="sweep-001")

    @pytest.mark.asyncio
    async def test_sets_tracking_uri_when_provided(self, mock_mlflow):
        async with OrcaTracker("exp", tracking_uri="http://mlflow:5000"):
            mock_mlflow.set_tracking_uri.assert_called_once_with("http://mlflow:5000")

    @pytest.mark.asyncio
    async def test_skips_tracking_uri_when_none(self, mock_mlflow):
        async with OrcaTracker("exp"):
            mock_mlflow.set_tracking_uri.assert_not_called()

    @pytest.mark.asyncio
    async def test_aenter_returns_tracker_instance(self, mock_mlflow):
        tracker = OrcaTracker("exp")
        result = await tracker.__aenter__()
        await tracker.__aexit__(None, None, None)
        assert result is tracker


class TestOrcaTrackerAexit:
    @pytest.mark.asyncio
    async def test_ends_run_on_clean_exit(self, mock_mlflow):
        async with OrcaTracker("exp"):
            pass
        mock_mlflow.end_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_ends_run_on_exception(self, mock_mlflow):
        with pytest.raises(ValueError):
            async with OrcaTracker("exp"):
                raise ValueError("training exploded")
        mock_mlflow.end_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_lifecycle_order(self, mock_mlflow):
        """set_experiment → start_run → (body) → end_run, in order."""
        call_log = []
        mock_mlflow.set_experiment.side_effect = lambda *a, **kw: call_log.append("set_experiment")
        mock_mlflow.start_run.side_effect = lambda *a, **kw: call_log.append("start_run")
        mock_mlflow.end_run.side_effect = lambda *a, **kw: call_log.append("end_run")

        async with OrcaTracker("exp"):
            call_log.append("body")

        assert call_log == ["set_experiment", "start_run", "body", "end_run"]


class TestOrcaTrackerLogging:
    @pytest.mark.asyncio
    async def test_log_params_delegates_to_mlflow(self, mock_mlflow):
        async with OrcaTracker("exp") as t:
            t.log_params({"lr": 0.001, "batch_size": 32})
        mock_mlflow.log_params.assert_called_once_with({"lr": 0.001, "batch_size": 32})

    @pytest.mark.asyncio
    async def test_log_metric_with_step(self, mock_mlflow):
        async with OrcaTracker("exp") as t:
            t.log_metric("accuracy", 0.9, step=5)
        mock_mlflow.log_metric.assert_called_once_with("accuracy", 0.9, step=5)

    @pytest.mark.asyncio
    async def test_log_metric_without_step(self, mock_mlflow):
        async with OrcaTracker("exp") as t:
            t.log_metric("loss", 0.3)
        mock_mlflow.log_metric.assert_called_once_with("loss", 0.3, step=None)

    @pytest.mark.asyncio
    async def test_log_artifact_delegates_to_mlflow(self, mock_mlflow):
        async with OrcaTracker("exp") as t:
            t.log_artifact("/tmp/model.pt")
        mock_mlflow.log_artifact.assert_called_once_with("/tmp/model.pt")

    @pytest.mark.asyncio
    async def test_multiple_params_logged(self, mock_mlflow):
        async with OrcaTracker("exp") as t:
            t.log_params({"a": 1})
            t.log_params({"b": 2})
        assert mock_mlflow.log_params.call_count == 2


# ---------------------------------------------------------------------------
# MetricLogger
# ---------------------------------------------------------------------------


class TestMetricLogger:
    def test_log_calls_mlflow_log_metric(self, mock_mlflow):
        MetricLogger().log("loss", 0.5)
        mock_mlflow.log_metric.assert_called_once_with("loss", 0.5, step=None, run_id=None)

    def test_log_passes_step(self, mock_mlflow):
        MetricLogger().log("acc", 0.9, step=3)
        mock_mlflow.log_metric.assert_called_once_with("acc", 0.9, step=3, run_id=None)

    def test_log_passes_run_id(self, mock_mlflow):
        MetricLogger(run_id="run-abc").log("f1", 0.8)
        mock_mlflow.log_metric.assert_called_once_with("f1", 0.8, step=None, run_id="run-abc")

    def test_log_batch_calls_log_for_each_metric(self, mock_mlflow):
        MetricLogger().log_batch({"acc": 0.9, "loss": 0.1})
        assert mock_mlflow.log_metric.call_count == 2
        called_names = {c.args[0] for c in mock_mlflow.log_metric.call_args_list}
        assert called_names == {"acc", "loss"}

    def test_log_batch_passes_step_to_all(self, mock_mlflow):
        MetricLogger().log_batch({"acc": 0.9, "f1": 0.85}, step=10)
        for c in mock_mlflow.log_metric.call_args_list:
            assert c.kwargs.get("step") == 10 or c.args[2] == 10

    def test_log_batch_empty_dict_no_calls(self, mock_mlflow):
        MetricLogger().log_batch({})
        mock_mlflow.log_metric.assert_not_called()

    def test_default_run_id_is_none(self, mock_mlflow):
        logger = MetricLogger()
        assert logger._run_id is None

    def test_run_id_stored(self, mock_mlflow):
        logger = MetricLogger(run_id="xyz")
        assert logger._run_id == "xyz"
