"""Unit tests for live_experiments page – pure data functions."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def le(_patch_streamlit):
    sys.modules.pop("orcalab.visualization.pages.live_experiments", None)
    sys.modules.pop("orcalab.visualization.components.metric_plots", None)
    return importlib.import_module("orcalab.visualization.pages.live_experiments")


EXPERIMENT_LIST = [
    {
        "experiment_id": "aaa-111",
        "status": "RUNNING",
        "current_epoch": 5,
        "total_epochs": 10,
    },
    {
        "experiment_id": "bbb-222",
        "status": "PENDING",
        "current_epoch": None,
        "total_epochs": None,
    },
    {
        "experiment_id": "ccc-333",
        "status": "FAILED",
        "current_epoch": 3,
        "total_epochs": 10,
    },
    {
        "experiment_id": "ddd-444",
        "status": "COMPLETED",
        "current_epoch": 10,
        "total_epochs": 10,
    },
]


# ── fetch_experiments ─────────────────────────────────────────────────────────


class TestFetchExperiments:
    def test_returns_list(self, le):
        mock_resp = MagicMock()
        mock_resp.json.return_value = EXPERIMENT_LIST
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp):
            result = le.fetch_experiments("http://localhost:8001")
        assert result == EXPERIMENT_LIST

    def test_url_targets_experiments_endpoint(self, le):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp) as mock_get:
            le.fetch_experiments("http://localhost:8001")
        url = mock_get.call_args.args[0]
        assert "/api/v1/experiments" in url

    def test_propagates_http_error(self, le):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("503")
        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(Exception, match="503"):
                le.fetch_experiments("http://localhost:8001")


# ── fetch_experiment_history ──────────────────────────────────────────────────


class TestFetchExperimentHistory:
    def test_returns_list(self, le):
        mock_resp = MagicMock()
        history = [{"epoch": 1, "loss": 0.5}]
        mock_resp.json.return_value = history
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp):
            result = le.fetch_experiment_history("http://localhost:8001", "aaa-111")
        assert result == history

    def test_url_contains_experiment_id(self, le):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp) as mock_get:
            le.fetch_experiment_history("http://localhost:8001", "my-exp-id")
        url = mock_get.call_args.args[0]
        assert "my-exp-id" in url

    def test_url_contains_history_segment(self, le):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp) as mock_get:
            le.fetch_experiment_history("http://localhost:8001", "x")
        url = mock_get.call_args.args[0]
        assert "history" in url


# ── color_for_status ──────────────────────────────────────────────────────────


class TestColorForStatus:
    def test_running_is_green(self, le):
        assert le.color_for_status("RUNNING") == "#28a745"

    def test_pending_is_gray(self, le):
        assert le.color_for_status("PENDING") == "#6c757d"

    def test_failed_is_red(self, le):
        assert le.color_for_status("FAILED") == "#dc3545"

    def test_completed_is_blue(self, le):
        assert le.color_for_status("COMPLETED") == "#007bff"

    def test_unknown_status_returns_gray(self, le):
        assert le.color_for_status("UNKNOWN_XYZ") == "#6c757d"

    def test_case_insensitive(self, le):
        assert le.color_for_status("running") == le.color_for_status("RUNNING")


# ── compute_progress ──────────────────────────────────────────────────────────


class TestComputeProgress:
    def test_negative_current_epoch_returns_zero(self, le):
        assert le.compute_progress(-1, 10) == 0.0

    def test_returns_correct_ratio(self, le):
        assert abs(le.compute_progress(5, 10) - 0.5) < 1e-9

    def test_clamps_to_one(self, le):
        assert le.compute_progress(15, 10) == 1.0

    def test_none_current_epoch_returns_zero(self, le):
        assert le.compute_progress(None, 10) == 0.0

    def test_none_total_epochs_returns_zero(self, le):
        assert le.compute_progress(5, None) == 0.0

    def test_zero_total_epochs_returns_zero(self, le):
        assert le.compute_progress(5, 0) == 0.0

    def test_full_completion(self, le):
        assert le.compute_progress(10, 10) == 1.0
