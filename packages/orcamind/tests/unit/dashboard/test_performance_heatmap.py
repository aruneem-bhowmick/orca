"""Unit tests for the performance heatmap page – pure data functions."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def ph(_patch_streamlit):
    sys.modules.pop("orcamind.dashboard.pages.performance_heatmap", None)
    mod = importlib.import_module("orcamind.dashboard.pages.performance_heatmap")
    return mod


PERF_RECORDS = [
    {"task_name": "iris", "architecture": "mlp", "mean_accuracy": 0.95},
    {"task_name": "iris", "architecture": "svm", "mean_accuracy": 0.92},
    {"task_name": "digits", "architecture": "mlp", "mean_accuracy": 0.98},
    {"task_name": "digits", "architecture": "svm", "mean_accuracy": 0.97},
    {"task_name": "boston", "architecture": "mlp", "mean_accuracy": 0.85},
    # boston/svm intentionally missing → will be NaN in heatmap
]


# ── fetch_performances ────────────────────────────────────────────────────────

class TestFetchPerformances:
    def test_returns_list_of_records(self, ph):
        mock_resp = MagicMock()
        mock_resp.json.return_value = PERF_RECORDS
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp):
            result = ph.fetch_performances("http://localhost:8000")
        assert result == PERF_RECORDS

    def test_passes_metric_name_param(self, ph):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp) as mock_get:
            ph.fetch_performances("http://localhost:8000", metric_name="val_accuracy")
        params = mock_get.call_args.kwargs.get("params", {})
        assert params.get("metric_name") == "val_accuracy"

    def test_default_metric_name_is_accuracy(self, ph):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp) as mock_get:
            ph.fetch_performances("http://localhost:8000")
        params = mock_get.call_args.kwargs.get("params", {})
        assert params.get("metric_name") == "accuracy"

    def test_propagates_http_error(self, ph):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("503")
        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(Exception, match="503"):
                ph.fetch_performances("http://localhost:8000")

    def test_url_targets_performances_endpoint(self, ph):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp) as mock_get:
            ph.fetch_performances("http://localhost:8000")
        url = mock_get.call_args.args[0]
        assert "/api/v1/performances" in url


# ── build_heatmap_df ──────────────────────────────────────────────────────────

class TestBuildHeatmapDf:
    def test_returns_empty_df_for_no_records(self, ph):
        result = ph.build_heatmap_df([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_pivot_rows_are_task_names(self, ph):
        result = ph.build_heatmap_df(PERF_RECORDS)
        assert set(result.index) == {"iris", "digits", "boston"}

    def test_pivot_columns_are_architectures(self, ph):
        result = ph.build_heatmap_df(PERF_RECORDS)
        assert set(result.columns) == {"mlp", "svm"}

    def test_known_cell_values(self, ph):
        result = ph.build_heatmap_df(PERF_RECORDS)
        assert abs(result.loc["iris", "mlp"] - 0.95) < 1e-6
        assert abs(result.loc["digits", "svm"] - 0.97) < 1e-6

    def test_missing_cell_is_nan(self, ph):
        result = ph.build_heatmap_df(PERF_RECORDS)
        assert pd.isna(result.loc["boston", "svm"])

    def test_duplicate_records_are_averaged(self, ph):
        dupes = [
            {"task_name": "a", "architecture": "x", "mean_accuracy": 0.8},
            {"task_name": "a", "architecture": "x", "mean_accuracy": 0.9},
        ]
        result = ph.build_heatmap_df(dupes)
        assert abs(result.loc["a", "x"] - 0.85) < 1e-6
