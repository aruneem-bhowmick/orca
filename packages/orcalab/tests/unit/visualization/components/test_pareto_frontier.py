"""Unit tests for pareto_frontier component."""

from __future__ import annotations

import importlib
import sys
from uuid import UUID

import pytest


@pytest.fixture(scope="module")
def pp(_patch_streamlit):
    sys.modules.pop("orcalab.visualization.components.pareto_frontier", None)
    return importlib.import_module("orcalab.visualization.components.pareto_frontier")


def _make_result(exp_id: str, metrics: dict | None):
    from orca_shared.schemas.training import ExperimentResult

    return ExperimentResult(
        experiment_id=UUID(exp_id),
        status="COMPLETED",
        metrics=metrics,
    )


# ── _is_pareto_optimal ────────────────────────────────────────────────────────


class TestIsParetoOptimal:
    def test_single_point_is_optimal(self, pp):
        result = pp._is_pareto_optimal([(0.5, 0.5)])
        assert result == [True]

    def test_dominated_point_is_not_optimal(self, pp):
        # (0.3, 0.3) dominates (0.5, 0.5) under minimisation
        result = pp._is_pareto_optimal([(0.5, 0.5), (0.3, 0.3)])
        assert result[0] is False
        assert result[1] is True

    def test_all_points_on_front_when_tradeoff(self, pp):
        # (0.1, 0.9) and (0.9, 0.1): neither dominates the other
        result = pp._is_pareto_optimal([(0.1, 0.9), (0.9, 0.1)])
        assert all(result)

    def test_empty_list_returns_empty(self, pp):
        assert pp._is_pareto_optimal([]) == []

    def test_mixed_domination(self, pp):
        # A=(0.2, 0.3): dominates C on both axes
        # B=(0.8, 0.2): non-dominated w.r.t. A (worse on x, better on y)
        # C=(0.5, 0.5): A dominates C (0.2≤0.5 AND 0.3≤0.5 with strict on both)
        result = pp._is_pareto_optimal([(0.2, 0.3), (0.8, 0.2), (0.5, 0.5)])
        assert result[0] is True   # A not dominated
        assert result[1] is True   # B not dominated
        assert result[2] is False  # C dominated by A


# ── pareto_plot ───────────────────────────────────────────────────────────────


class TestParetoPlot:
    def test_empty_results_does_not_raise(self, pp):
        fig = pp.pareto_plot([], "loss", "latency")
        assert fig is not None

    def test_skips_experiments_with_no_metrics(self, pp):
        results = [_make_result("00000000-0000-0000-0000-000000000001", None)]
        fig = pp.pareto_plot(results, "loss", "latency")
        assert fig is not None

    def test_skips_experiments_missing_x_metric(self, pp):
        results = [
            _make_result("00000000-0000-0000-0000-000000000001", {"latency": 0.5})
        ]
        fig = pp.pareto_plot(results, "loss", "latency")
        assert fig is not None

    def test_skips_nan_metric_values(self, pp):
        results = [
            _make_result(
                "00000000-0000-0000-0000-000000000001",
                {"loss": float("nan"), "latency": 0.5},
            )
        ]
        fig = pp.pareto_plot(results, "loss", "latency")
        assert fig is not None

    def test_returns_figure_for_valid_results(self, pp):
        results = [
            _make_result(
                "00000000-0000-0000-0000-000000000001", {"loss": 0.2, "latency": 0.8}
            ),
            _make_result(
                "00000000-0000-0000-0000-000000000002", {"loss": 0.8, "latency": 0.2}
            ),
        ]
        fig = pp.pareto_plot(results, "loss", "latency")
        assert fig is not None

    def test_pareto_optimal_points_rendered(self, pp):
        """Two non-dominated points; both should generate a Pareto-optimal trace."""
        results = [
            _make_result(
                "00000000-0000-0000-0000-000000000001", {"loss": 0.2, "latency": 0.8}
            ),
            _make_result(
                "00000000-0000-0000-0000-000000000002", {"loss": 0.8, "latency": 0.2}
            ),
        ]
        fig = pp.pareto_plot(results, "loss", "latency")
        # add_trace called at least once for Pareto-optimal trace (red diamonds)
        assert fig.add_trace.called
