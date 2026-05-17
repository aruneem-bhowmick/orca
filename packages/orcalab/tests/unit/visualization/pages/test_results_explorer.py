"""Unit tests for results_explorer page – pure data functions."""

from __future__ import annotations

import importlib
import sys
from datetime import date
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def re(_patch_streamlit):
    sys.modules.pop("orcalab.visualization.pages.results_explorer", None)
    sys.modules.pop("orcalab.visualization.components.metric_plots", None)
    return importlib.import_module("orcalab.visualization.pages.results_explorer")


EXPERIMENTS = [
    {
        "experiment_id": "exp-1",
        "task_id": "task-a",
        "domain": "vision",
        "completed_at": "2024-03-15T12:00:00",
        "metrics": {"accuracy": 0.92},
    },
    {
        "experiment_id": "exp-2",
        "task_id": "task-b",
        "domain": "nlp",
        "completed_at": "2024-04-01T08:00:00",
        "metrics": {"accuracy": 0.88},
    },
    {
        "experiment_id": "exp-3",
        "task_id": "task-a",
        "domain": "vision",
        "completed_at": "2024-02-10T06:00:00",
        "metrics": {"accuracy": 0.85},
    },
]


# ── fetch_completed_experiments ───────────────────────────────────────────────


class TestFetchCompletedExperiments:
    def test_returns_list(self, re):
        mock_resp = MagicMock()
        mock_resp.json.return_value = EXPERIMENTS
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp):
            result = re.fetch_completed_experiments("http://localhost:8001")
        assert result == EXPERIMENTS

    def test_url_targets_experiments_endpoint(self, re):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp) as mock_get:
            re.fetch_completed_experiments("http://localhost:8001")
        url = mock_get.call_args.args[0]
        assert "/api/v1/experiments" in url

    def test_passes_status_completed_param(self, re):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp) as mock_get:
            re.fetch_completed_experiments("http://localhost:8001")
        params = mock_get.call_args.kwargs.get("params", {})
        assert params.get("status") == "COMPLETED"

    def test_propagates_http_error(self, re):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("500")
        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(Exception, match="500"):
                re.fetch_completed_experiments("http://localhost:8001")


# ── filter_experiments ────────────────────────────────────────────────────────


class TestFilterExperiments:
    def test_no_filters_returns_all(self, re):
        result = re.filter_experiments(EXPERIMENTS)
        assert len(result) == 3

    def test_task_id_filter(self, re):
        result = re.filter_experiments(EXPERIMENTS, task_id="task-a")
        assert all(e["task_id"] == "task-a" for e in result)
        assert len(result) == 2

    def test_domain_filter(self, re):
        result = re.filter_experiments(EXPERIMENTS, domain="nlp")
        assert len(result) == 1
        assert result[0]["experiment_id"] == "exp-2"

    def test_date_from_filter(self, re):
        result = re.filter_experiments(
            EXPERIMENTS, date_from=date(2024, 3, 1)
        )
        assert all(e["completed_at"] >= "2024-03-01" for e in result)
        assert len(result) == 2

    def test_date_to_filter(self, re):
        result = re.filter_experiments(
            EXPERIMENTS, date_to=date(2024, 3, 31)
        )
        assert len(result) == 2

    def test_date_range_filter(self, re):
        result = re.filter_experiments(
            EXPERIMENTS,
            date_from=date(2024, 3, 1),
            date_to=date(2024, 3, 31),
        )
        assert len(result) == 1
        assert result[0]["experiment_id"] == "exp-1"

    def test_combined_task_and_domain_filter(self, re):
        result = re.filter_experiments(
            EXPERIMENTS, task_id="task-a", domain="vision"
        )
        assert len(result) == 2
        assert all(e["domain"] == "vision" for e in result)

    def test_experiment_missing_completed_at_excluded_when_date_filter_active(self, re):
        exps = [{"experiment_id": "x", "task_id": "t", "domain": "d"}]
        result = re.filter_experiments(exps, date_from=date(2024, 1, 1))
        assert result == []

    def test_invalid_date_string_excluded(self, re):
        exps = [{"experiment_id": "x", "completed_at": "not-a-date"}]
        result = re.filter_experiments(exps, date_from=date(2024, 1, 1))
        assert result == []

    def test_empty_input_returns_empty(self, re):
        assert re.filter_experiments([]) == []


# ── diff_configs ──────────────────────────────────────────────────────────────


class TestDiffConfigs:
    def test_no_diff_for_identical_dicts(self, re):
        exp = {"a": 1, "b": 2}
        assert re.diff_configs(exp, exp) == {}

    def test_detects_value_differences(self, re):
        a = {"lr": 0.001}
        b = {"lr": 0.01}
        diff = re.diff_configs(a, b)
        assert "lr" in diff
        assert diff["lr"]["a"] == 0.001
        assert diff["lr"]["b"] == 0.01

    def test_detects_keys_present_only_in_a(self, re):
        a = {"lr": 0.001, "extra": "yes"}
        b = {"lr": 0.001}
        diff = re.diff_configs(a, b)
        assert "extra" in diff
        assert diff["extra"]["b"] is None

    def test_detects_keys_present_only_in_b(self, re):
        a = {"lr": 0.001}
        b = {"lr": 0.001, "extra": "yes"}
        diff = re.diff_configs(a, b)
        assert "extra" in diff
        assert diff["extra"]["a"] is None

    def test_result_keys_are_sorted(self, re):
        a = {"z": 1, "a": 2}
        b = {"z": 9, "a": 2}
        diff = re.diff_configs(a, b)
        assert list(diff.keys()) == sorted(diff.keys())
