"""Unit tests for the training progress page – pure data functions."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def tp(_patch_streamlit):
    sys.modules.pop("orcamind.dashboard.pages.training_progress", None)
    # mlflow must be importable; mock it if not installed.
    if "mlflow" not in sys.modules:
        sys.modules["mlflow"] = MagicMock()
        sys.modules["mlflow.tracking"] = MagicMock()
    mod = importlib.import_module("orcamind.dashboard.pages.training_progress")
    return mod


# ── fetch_mlflow_runs ─────────────────────────────────────────────────────────

class TestFetchMlflowRuns:
    def test_returns_dataframe_on_success(self, tp):
        expected = pd.DataFrame({"run_id": ["r1", "r2"], "status": ["FINISHED", "RUNNING"]})
        with patch("mlflow.search_runs", return_value=expected):
            with patch("mlflow.set_tracking_uri"):
                result = tp.fetch_mlflow_runs("http://localhost:5000")
        assert isinstance(result, pd.DataFrame)
        assert list(result["run_id"]) == ["r1", "r2"]

    def test_sets_tracking_uri(self, tp):
        with patch("mlflow.search_runs", return_value=pd.DataFrame()):
            with patch("mlflow.set_tracking_uri") as mock_uri:
                tp.fetch_mlflow_runs("http://mlflow:5000")
        mock_uri.assert_called_once_with("http://mlflow:5000")

    def test_returns_empty_df_on_error(self, tp):
        with patch("mlflow.search_runs", side_effect=Exception("connection refused")):
            with patch("mlflow.set_tracking_uri"):
                result = tp.fetch_mlflow_runs("http://bad-host")
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ── build_metric_df ───────────────────────────────────────────────────────────

class TestBuildMetricDf:
    def _make_metric(self, step: int, value: float):
        m = MagicMock()
        m.step = step
        m.value = value
        return m

    def test_returns_per_epoch_records(self, tp):
        client = MagicMock()
        client.get_metric_history.return_value = [
            self._make_metric(1, 0.5),
            self._make_metric(2, 0.4),
            self._make_metric(3, 0.3),
        ]
        result = tp.build_metric_df(client, "meta_train_loss", ["run_abc"])
        assert len(result) == 3
        assert list(result["epoch"]) == [1, 2, 3]
        assert list(result["value"]) == [0.5, 0.4, 0.3]

    def test_combines_multiple_runs(self, tp):
        def side_effect(run_id, metric):
            return [self._make_metric(1, 0.9)] if run_id == "r1" else [self._make_metric(1, 0.8)]

        client = MagicMock()
        client.get_metric_history.side_effect = side_effect
        result = tp.build_metric_df(client, "meta_train_accuracy", ["r1", "r2"])
        assert len(result) == 2
        assert set(result["run_id"]) == {"r1", "r2"}

    def test_returns_empty_df_when_no_history(self, tp):
        client = MagicMock()
        client.get_metric_history.return_value = []
        result = tp.build_metric_df(client, "meta_train_loss", ["run_none"])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_skips_erroring_run(self, tp):
        client = MagicMock()
        client.get_metric_history.side_effect = Exception("run not found")
        result = tp.build_metric_df(client, "meta_train_loss", ["bad_run"])
        assert result.empty

    def test_columns_are_present_in_empty_result(self, tp):
        client = MagicMock()
        client.get_metric_history.return_value = []
        result = tp.build_metric_df(client, "meta_train_loss", [])
        assert set(result.columns) >= {"run_id", "epoch", "value"}
