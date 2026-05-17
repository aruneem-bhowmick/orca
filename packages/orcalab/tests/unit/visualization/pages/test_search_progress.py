"""Unit tests for search_progress page – pure data functions."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def sp(_patch_streamlit):
    sys.modules.pop("orcalab.visualization.pages.search_progress", None)
    sys.modules.pop("orcalab.visualization.components.parallel_coords", None)
    return importlib.import_module("orcalab.visualization.pages.search_progress")


SWEEP_LIST = [
    {"sweep_id": "sweep-1", "name": "lr_search"},
    {"sweep_id": "sweep-2", "name": "depth_search"},
]

TRIAL_LIST = [
    {"lr": 0.001, "objective": 0.92},
    {"lr": 0.01, "objective": 0.87},
    {"lr": 0.0001, "objective": 0.95},
]


# ── fetch_sweeps ──────────────────────────────────────────────────────────────


class TestFetchSweeps:
    def test_returns_list(self, sp):
        mock_resp = MagicMock()
        mock_resp.json.return_value = SWEEP_LIST
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp):
            result = sp.fetch_sweeps("http://localhost:8001")
        assert result == SWEEP_LIST

    def test_url_targets_sweeps_endpoint(self, sp):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp) as mock_get:
            sp.fetch_sweeps("http://localhost:8001")
        url = mock_get.call_args.args[0]
        assert "/api/v1/sweeps" in url

    def test_propagates_http_error(self, sp):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404")
        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(Exception, match="404"):
                sp.fetch_sweeps("http://localhost:8001")


# ── fetch_sweep_trials ────────────────────────────────────────────────────────


class TestFetchSweepTrials:
    def test_returns_trials(self, sp):
        mock_resp = MagicMock()
        mock_resp.json.return_value = TRIAL_LIST
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp):
            result = sp.fetch_sweep_trials("http://localhost:8001", "sweep-1")
        assert result == TRIAL_LIST

    def test_url_contains_sweep_id(self, sp):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp) as mock_get:
            sp.fetch_sweep_trials("http://localhost:8001", "sweep-xyz")
        url = mock_get.call_args.args[0]
        assert "sweep-xyz" in url

    def test_url_contains_trials_segment(self, sp):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp) as mock_get:
            sp.fetch_sweep_trials("http://localhost:8001", "x")
        url = mock_get.call_args.args[0]
        assert "trials" in url


# ── find_best_trial ───────────────────────────────────────────────────────────


class TestFindBestTrial:
    def test_returns_none_for_empty(self, sp):
        assert sp.find_best_trial([]) is None

    def test_returns_trial_with_highest_objective(self, sp):
        best = sp.find_best_trial(TRIAL_LIST)
        assert best is not None
        assert best["objective"] == 0.95

    def test_returns_none_when_all_objectives_missing(self, sp):
        trials = [{"lr": 0.001}, {"lr": 0.01}]
        assert sp.find_best_trial(trials) is None

    def test_skips_none_objectives(self, sp):
        trials = [
            {"lr": 0.001, "objective": None},
            {"lr": 0.01, "objective": 0.87},
        ]
        best = sp.find_best_trial(trials)
        assert best is not None
        assert best["objective"] == pytest.approx(0.87)

    def test_skips_non_numeric_objectives(self, sp):
        trials = [
            {"lr": 0.001, "objective": "bad"},
            {"lr": 0.01, "objective": 0.87},
        ]
        best = sp.find_best_trial(trials)
        assert best is not None
        assert best["objective"] == pytest.approx(0.87)


# ── build_cumulative_df ───────────────────────────────────────────────────────


class TestBuildCumulativeDf:
    def test_empty_trials_returns_empty_df(self, sp):
        df = sp.build_cumulative_df([])
        assert df.empty

    def test_empty_df_has_expected_columns(self, sp):
        df = sp.build_cumulative_df([])
        assert set(df.columns) == {"trial_index", "cumulative_count"}

    def test_cumulative_count_increments(self, sp):
        df = sp.build_cumulative_df(TRIAL_LIST)
        assert list(df["cumulative_count"]) == [1, 2, 3]

    def test_trial_index_starts_at_one(self, sp):
        df = sp.build_cumulative_df(TRIAL_LIST)
        assert df["trial_index"].iloc[0] == 1

    def test_row_count_matches_trial_count(self, sp):
        df = sp.build_cumulative_df(TRIAL_LIST)
        assert len(df) == len(TRIAL_LIST)
