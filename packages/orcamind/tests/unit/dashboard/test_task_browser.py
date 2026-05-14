"""Unit tests for the task browser page – pure data functions."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def tb(_patch_streamlit):
    """Import task_browser after streamlit has been mocked."""
    sys.modules.pop("orcamind.dashboard.pages.task_browser", None)
    mod = importlib.import_module("orcamind.dashboard.pages.task_browser")
    return mod


TASK_LIST = [
    {
        "task_id": "aaaa-1111",
        "name": "iris",
        "domain": "biology",
        "task_type": "classification",
        "n_samples": 150,
        "n_features": 4,
        "n_classes": 3,
    },
    {
        "task_id": "bbbb-2222",
        "name": "digits",
        "domain": "vision",
        "task_type": "classification",
        "n_samples": 1797,
        "n_features": 64,
        "n_classes": 10,
    },
    {
        "task_id": "cccc-3333",
        "name": "boston",
        "domain": "economics",
        "task_type": "regression",
        "n_samples": 506,
        "n_features": 13,
        "n_classes": None,
    },
]


# ── fetch_tasks ───────────────────────────────────────────────────────────────

class TestFetchTasks:
    def test_returns_task_list(self, tb):
        mock_resp = MagicMock()
        mock_resp.json.return_value = TASK_LIST
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp) as mock_get:
            result = tb.fetch_tasks("http://localhost:8000")
        assert result == TASK_LIST
        mock_get.assert_called_once_with(
            "http://localhost:8000/api/v1/tasks", params={}, timeout=10
        )

    def test_passes_domain_filter(self, tb):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp) as mock_get:
            tb.fetch_tasks("http://localhost:8000", domain="vision")
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["domain"] == "vision"

    def test_passes_task_type_filter(self, tb):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp) as mock_get:
            tb.fetch_tasks("http://localhost:8000", task_type="regression")
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["task_type"] == "regression"

    def test_propagates_http_error(self, tb):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404")
        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(Exception, match="404"):
                tb.fetch_tasks("http://bad-host")


# ── fetch_task_detail ─────────────────────────────────────────────────────────

class TestFetchTaskDetail:
    def test_returns_single_task(self, tb):
        mock_resp = MagicMock()
        mock_resp.json.return_value = TASK_LIST[0]
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp):
            result = tb.fetch_task_detail("http://localhost:8000", "aaaa-1111")
        assert result["task_id"] == "aaaa-1111"

    def test_url_contains_task_id(self, tb):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp) as mock_get:
            tb.fetch_task_detail("http://localhost:8000", "xyz-999")
        url = mock_get.call_args.args[0]
        assert "xyz-999" in url


# ── build_pca_df ──────────────────────────────────────────────────────────────

class TestBuildPcaDf:
    def test_returns_none_for_single_task(self, tb):
        assert tb.build_pca_df([TASK_LIST[0]], "aaaa-1111") is None

    def test_returns_dataframe_for_two_tasks(self, tb):
        result = tb.build_pca_df(TASK_LIST[:2], "aaaa-1111")
        assert isinstance(result, pd.DataFrame)
        assert "PC1" in result.columns
        assert "PC2" in result.columns

    def test_selected_flag_is_set(self, tb):
        result = tb.build_pca_df(TASK_LIST, "aaaa-1111")
        assert result is not None
        selected_rows = result[result["selected"]]
        assert len(selected_rows) == 1
        assert selected_rows.iloc[0]["task_id"] == "aaaa-1111"

    def test_non_selected_tasks_are_false(self, tb):
        result = tb.build_pca_df(TASK_LIST, "aaaa-1111")
        non_selected = result[~result["selected"]]
        assert len(non_selected) == len(TASK_LIST) - 1

    def test_handles_none_numeric_fields(self, tb):
        tasks_with_none = [
            {"task_id": "t1", "name": "a", "n_samples": None, "n_features": None, "n_classes": None},
            {"task_id": "t2", "name": "b", "n_samples": None, "n_features": None, "n_classes": None},
        ]
        result = tb.build_pca_df(tasks_with_none, "t1")
        assert result is not None
        assert len(result) == 2
