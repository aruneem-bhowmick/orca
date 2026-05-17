"""Unit tests for meta_analysis page – pure data functions."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def ma(_patch_streamlit):
    sys.modules.pop("orcalab.visualization.pages.meta_analysis", None)
    return importlib.import_module("orcalab.visualization.pages.meta_analysis")


EXPERIMENTS = [
    {
        "experiment_id": "e1",
        "domain": "vision",
        "architecture": "cnn",
        "n_features": 64,
        "n_samples": 1000,
        "completed_at": "2024-01-10T12:00:00",
        "metrics": {"accuracy": 0.90},
    },
    {
        "experiment_id": "e2",
        "domain": "vision",
        "architecture": "mlp",
        "n_features": 64,
        "n_samples": 1000,
        "completed_at": "2024-01-12T12:00:00",
        "metrics": {"accuracy": 0.85},
    },
    {
        "experiment_id": "e3",
        "domain": "nlp",
        "architecture": "transformer",
        "n_features": 128,
        "n_samples": 2000,
        "completed_at": "2024-01-15T12:00:00",
        "metrics": {"accuracy": 0.95},
    },
]


# ── fetch_all_experiments ─────────────────────────────────────────────────────


class TestFetchAllExperiments:
    def test_returns_list(self, ma):
        mock_resp = MagicMock()
        mock_resp.json.return_value = EXPERIMENTS
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp):
            result = ma.fetch_all_experiments("http://localhost:8001")
        assert result == EXPERIMENTS

    def test_url_targets_experiments_endpoint(self, ma):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp) as mock_get:
            ma.fetch_all_experiments("http://localhost:8001")
        url = mock_get.call_args.args[0]
        assert "/api/v1/experiments" in url

    def test_propagates_http_error(self, ma):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("503")
        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(Exception, match="503"):
                ma.fetch_all_experiments("http://localhost:8001")


# ── build_domain_arch_heatmap ─────────────────────────────────────────────────


class TestBuildDomainArchHeatmap:
    def test_returns_empty_df_for_empty_input(self, ma):
        result = ma.build_domain_arch_heatmap([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_pivot_rows_are_domains(self, ma):
        result = ma.build_domain_arch_heatmap(EXPERIMENTS)
        assert set(result.index) == {"vision", "nlp"}

    def test_pivot_columns_are_architectures(self, ma):
        result = ma.build_domain_arch_heatmap(EXPERIMENTS)
        assert "cnn" in result.columns
        assert "mlp" in result.columns
        assert "transformer" in result.columns

    def test_known_cell_value(self, ma):
        result = ma.build_domain_arch_heatmap(EXPERIMENTS)
        assert abs(result.loc["vision", "cnn"] - 0.90) < 1e-9

    def test_missing_combination_is_nan(self, ma):
        result = ma.build_domain_arch_heatmap(EXPERIMENTS)
        # nlp/cnn combination does not exist in the data
        assert pd.isna(result.loc["nlp", "cnn"])

    def test_excludes_experiments_missing_domain(self, ma):
        exps = [{"architecture": "mlp", "metrics": {"accuracy": 0.9}}]
        result = ma.build_domain_arch_heatmap(exps)
        assert result.empty

    def test_excludes_experiments_missing_metric(self, ma):
        exps = [{"domain": "vision", "architecture": "mlp", "metrics": {}}]
        result = ma.build_domain_arch_heatmap(exps)
        assert result.empty

    def test_multiple_records_same_cell_are_averaged(self, ma):
        exps = [
            {"domain": "d", "architecture": "a", "metrics": {"accuracy": 0.8}},
            {"domain": "d", "architecture": "a", "metrics": {"accuracy": 0.9}},
        ]
        result = ma.build_domain_arch_heatmap(exps)
        assert abs(result.loc["d", "a"] - 0.85) < 1e-9


# ── build_scatter_df ──────────────────────────────────────────────────────────


class TestBuildScatterDf:
    def test_returns_empty_df_for_empty_input(self, ma):
        result = ma.build_scatter_df([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_empty_df_has_expected_columns(self, ma):
        result = ma.build_scatter_df([])
        assert set(result.columns) == {"complexity", "accuracy", "experiment_id"}

    def test_complexity_is_n_features_times_n_samples(self, ma):
        result = ma.build_scatter_df(EXPERIMENTS)
        vision_rows = result[result["experiment_id"] == "e1"]
        assert len(vision_rows) == 1
        assert vision_rows.iloc[0]["complexity"] == 64 * 1000

    def test_excludes_experiments_without_complexity_data(self, ma):
        exps = [{"experiment_id": "x", "metrics": {"accuracy": 0.9}}]
        result = ma.build_scatter_df(exps)
        assert result.empty

    def test_excludes_experiments_without_metric(self, ma):
        exps = [{"experiment_id": "x", "n_features": 10, "n_samples": 100, "metrics": {}}]
        result = ma.build_scatter_df(exps)
        assert result.empty

    def test_excludes_experiment_with_only_one_complexity_dimension(self, ma):
        exps = [{"experiment_id": "x", "n_features": 10, "metrics": {"accuracy": 0.9}}]
        result = ma.build_scatter_df(exps)
        assert result.empty


# ── build_trend_df ────────────────────────────────────────────────────────────


class TestBuildTrendDf:
    def test_returns_empty_df_for_empty_input(self, ma):
        result = ma.build_trend_df([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_empty_df_has_expected_columns(self, ma):
        result = ma.build_trend_df([])
        assert set(result.columns) == {"completed_at", "value", "best_so_far"}

    def test_best_so_far_is_cumulative_max(self, ma):
        result = ma.build_trend_df(EXPERIMENTS)
        # Sorted by date: 0.90, 0.85, 0.95 → cummax: 0.90, 0.90, 0.95
        best_vals = result["best_so_far"].tolist()
        assert best_vals[0] == pytest.approx(0.90)
        assert best_vals[1] == pytest.approx(0.90)
        assert best_vals[2] == pytest.approx(0.95)

    def test_sorted_by_completed_at(self, ma):
        import pandas as pd

        result = ma.build_trend_df(EXPERIMENTS)
        ts = result["completed_at"].tolist()
        assert ts == sorted(ts)

    def test_excludes_experiments_without_timestamp(self, ma):
        exps = [{"experiment_id": "x", "metrics": {"accuracy": 0.9}}]
        result = ma.build_trend_df(exps)
        assert result.empty

    def test_excludes_experiments_without_metric(self, ma):
        exps = [{"experiment_id": "x", "completed_at": "2024-01-01T00:00:00", "metrics": {}}]
        result = ma.build_trend_df(exps)
        assert result.empty

    def test_invalid_timestamp_is_excluded(self, ma):
        exps = [{"experiment_id": "x", "completed_at": "not-a-date", "metrics": {"accuracy": 0.9}}]
        result = ma.build_trend_df(exps)
        assert result.empty

    def test_row_count_matches_valid_experiments(self, ma):
        result = ma.build_trend_df(EXPERIMENTS)
        assert len(result) == len(EXPERIMENTS)
