"""Pareto frontier scatter plot."""

from __future__ import annotations

import math

import plotly.graph_objects as go

from orca_shared.schemas.training import ExperimentResult


def _is_pareto_optimal(costs: list[tuple[float, float]]) -> list[bool]:
    """Return a boolean mask of Pareto-optimal points (minimisation on both axes).

    Point ``i`` is dominated when some point ``j`` satisfies ``j <= i`` on both
    axes and ``j < i`` on at least one axis.
    """
    n = len(costs)
    is_optimal = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            xi, yi = costs[i]
            xj, yj = costs[j]
            if xj <= xi and yj <= yi and (xj < xi or yj < yi):
                is_optimal[i] = False
                break
    return is_optimal


def pareto_plot(
    results: list[ExperimentResult],
    x_metric: str,
    y_metric: str,
) -> go.Figure:
    """Scatter plot highlighting Pareto-optimal points in red (minimisation).

    Experiments with missing or ``NaN`` values for either metric axis are
    excluded.  All points on the Pareto frontier are shown as red diamonds;
    sub-optimal points are shown as steelblue circles.
    """
    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []

    for r in results:
        if not r.metrics:
            continue
        x_val = r.metrics.get(x_metric)
        y_val = r.metrics.get(y_metric)
        if x_val is None or y_val is None:
            continue
        try:
            fx, fy = float(x_val), float(y_val)
        except (TypeError, ValueError):
            continue
        if math.isnan(fx) or math.isnan(fy):
            continue
        xs.append(fx)
        ys.append(fy)
        labels.append(str(r.experiment_id)[:8])

    fig = go.Figure()
    fig.update_layout(
        title=f"Pareto Frontier: {x_metric} vs {y_metric}",
        xaxis_title=x_metric,
        yaxis_title=y_metric,
    )

    if not xs:
        return fig

    optimal_mask = _is_pareto_optimal(list(zip(xs, ys)))

    non_xs = [xs[i] for i in range(len(xs)) if not optimal_mask[i]]
    non_ys = [ys[i] for i in range(len(ys)) if not optimal_mask[i]]
    non_labels = [labels[i] for i in range(len(labels)) if not optimal_mask[i]]
    if non_xs:
        fig.add_trace(
            go.Scatter(
                x=non_xs,
                y=non_ys,
                mode="markers",
                name="Sub-optimal",
                marker=dict(color="steelblue", size=8),
                text=non_labels,
                hovertemplate="%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<extra></extra>",
            )
        )

    par_xs = [xs[i] for i in range(len(xs)) if optimal_mask[i]]
    par_ys = [ys[i] for i in range(len(ys)) if optimal_mask[i]]
    par_labels = [labels[i] for i in range(len(labels)) if optimal_mask[i]]
    if par_xs:
        fig.add_trace(
            go.Scatter(
                x=par_xs,
                y=par_ys,
                mode="markers",
                name="Pareto-optimal",
                marker=dict(color="red", size=10, symbol="diamond"),
                text=par_labels,
                hovertemplate="%{text}<br>x=%{x:.4f}<br>y=%{y:.4f}<extra></extra>",
            )
        )

    return fig
