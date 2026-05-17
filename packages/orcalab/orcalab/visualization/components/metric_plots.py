"""Loss curve and metric comparison bar chart components."""

from __future__ import annotations

import math

import plotly.graph_objects as go

from orca_shared.schemas.training import ExperimentResult


def loss_curve(history: list[dict], title: str = "Training Loss") -> go.Figure:
    """Line chart with one series per metric key found in the history dicts.

    Each dict must have an ``epoch`` key (or ``step`` as fallback) plus one or
    more float-valued metric keys.  Rows where the epoch or a metric value is
    ``None`` or ``NaN`` are skipped for that individual series.
    """
    fig = go.Figure()
    if not history:
        fig.update_layout(title=title)
        return fig

    epoch_key = "epoch" if any("epoch" in row for row in history) else "step"
    skip_keys = {epoch_key, "run_id"}
    # Union across all rows so metrics that first appear in later entries are included.
    metric_keys = sorted({k for row in history for k in row if k not in skip_keys})

    for key in metric_keys:
        epochs, values = [], []
        for row in history:
            ep = row.get(epoch_key)
            val = row.get(key)
            if ep is None or val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if not math.isnan(fval):
                epochs.append(ep)
                values.append(fval)
        fig.add_trace(go.Scatter(x=epochs, y=values, mode="lines", name=key))

    fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title="Value")
    return fig


def metric_comparison(results: list[ExperimentResult], metric: str) -> go.Figure:
    """Bar chart comparing the final ``metric`` value across experiments.

    Experiments with no ``metrics`` dict, or where the metric is absent or
    ``NaN``, are silently excluded.  Labels are the first 8 characters of the
    experiment UUID.
    """
    labels: list[str] = []
    values: list[float] = []

    for r in results:
        if not r.metrics:
            continue
        val = r.metrics.get(metric)
        if val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if math.isnan(fval):
            continue
        labels.append(str(r.experiment_id)[:8])
        values.append(fval)

    fig = go.Figure()
    if labels:
        fig.add_trace(go.Bar(x=labels, y=values, name=metric))
    fig.update_layout(
        title=f"{metric} comparison",
        xaxis_title="Experiment",
        yaxis_title=metric,
    )
    return fig
