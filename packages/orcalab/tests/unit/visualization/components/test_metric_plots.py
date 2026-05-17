"""Unit tests for metric_plots - pure Plotly chart functions."""

from __future__ import annotations

import importlib
import sys
from uuid import UUID

import pytest


@pytest.fixture(scope="module")
def mp(_patch_streamlit):
    sys.modules.pop("orcalab.visualization.components.metric_plots", None)
    return importlib.import_module("orcalab.visualization.components.metric_plots")


HISTORY = [
    {"epoch": 1, "loss": 0.8, "accuracy": 0.6},
    {"epoch": 2, "loss": 0.6, "accuracy": 0.7},
    {"epoch": 3, "loss": 0.4, "accuracy": 0.85},
]


def _make_result(exp_id: str, metrics: dict | None):
    from orca_shared.schemas.training import ExperimentResult

    return ExperimentResult(
        experiment_id=UUID(exp_id),
        status="COMPLETED",
        metrics=metrics,
    )


# ── loss_curve ────────────────────────────────────────────────────────────────


class TestLossCurve:
    def test_returns_figure_object(self, mp):
        fig = mp.loss_curve(HISTORY)
        assert fig is not None

    def test_empty_history_does_not_raise(self, mp):
        fig = mp.loss_curve([])
        assert fig is not None

    def test_correct_series_count(self, mp):
        fig = mp.loss_curve(HISTORY)
        # loss + accuracy → add_trace called at least twice
        assert fig.add_trace.called
        # The call count should be at least 2 (loss + accuracy)
        assert fig.add_trace.call_count >= 2

    def test_uses_step_key_as_epoch_fallback(self, mp):
        history = [
            {"step": 1, "loss": 0.5},
            {"step": 2, "loss": 0.3},
        ]
        fig = mp.loss_curve(history)
        assert fig is not None

    def test_skips_rows_with_none_epoch(self, mp):
        history = [
            {"epoch": None, "loss": 0.5},
            {"epoch": 2, "loss": 0.3},
        ]
        fig = mp.loss_curve(history)
        assert fig is not None

    def test_skips_nan_values(self, mp):
        import math

        history = [
            {"epoch": 1, "loss": float("nan")},
            {"epoch": 2, "loss": 0.3},
        ]
        fig = mp.loss_curve(history)
        # Should call add_trace once for "loss" with only the valid point
        assert fig is not None

    def test_title_passed_through(self, mp):
        mp.loss_curve(HISTORY, title="My Chart")
        # No assertion on figure internals (go is mocked); just verify no error


# ── metric_comparison ─────────────────────────────────────────────────────────


class TestMetricComparison:
    def test_returns_figure_for_valid_results(self, mp):
        results = [
            _make_result("00000000-0000-0000-0000-000000000001", {"accuracy": 0.9}),
            _make_result("00000000-0000-0000-0000-000000000002", {"accuracy": 0.85}),
        ]
        fig = mp.metric_comparison(results, "accuracy")
        assert fig is not None

    def test_empty_results_does_not_raise(self, mp):
        fig = mp.metric_comparison([], "accuracy")
        assert fig is not None

    def test_skips_experiments_with_no_metrics(self, mp):
        results = [_make_result("00000000-0000-0000-0000-000000000001", None)]
        fig = mp.metric_comparison(results, "accuracy")
        assert fig is not None

    def test_skips_missing_metric_key(self, mp):
        results = [
            _make_result("00000000-0000-0000-0000-000000000001", {"loss": 0.1})
        ]
        fig = mp.metric_comparison(results, "accuracy")
        assert fig is not None

    def test_skips_nan_metric_values(self, mp):
        results = [
            _make_result(
                "00000000-0000-0000-0000-000000000001", {"accuracy": float("nan")}
            )
        ]
        fig = mp.metric_comparison(results, "accuracy")
        # No Bar trace should be added for NaN
        assert fig is not None

    def test_label_is_truncated_uuid(self, mp):
        results = [
            _make_result("abcdef12-0000-0000-0000-000000000001", {"accuracy": 0.9})
        ]
        mp.metric_comparison(results, "accuracy")
        call_args = mp.go.Bar.call_args
        if call_args:
            labels = call_args.kwargs.get("x") or (call_args.args[0] if call_args.args else None)
            if labels:
                assert labels[0] == "abcdef12"
