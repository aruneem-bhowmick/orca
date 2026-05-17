"""Parallel coordinates chart for hyperparameter sweep visualisation."""

from __future__ import annotations

import plotly.graph_objects as go


def parallel_coordinates(
    trials: list[dict],
    colorscale: str = "Viridis",
) -> go.Figure:
    """Parallel coordinates plot coloured by the ``objective`` field.

    Each dict in ``trials`` should contain an ``"objective"`` key (float) plus
    one key per hyperparameter.  Non-numeric (categorical) parameter columns
    are mapped to integer codes so Plotly can render them; the original values
    appear as tick labels.  Missing objectives are treated as ``NaN`` (Plotly
    renders them in a neutral colour).
    """
    fig = go.Figure()
    if not trials:
        return fig

    param_keys = [k for k in trials[0] if k != "objective"]
    objective_vals = [
        float(t["objective"]) if t.get("objective") is not None else float("nan")
        for t in trials
    ]

    dimensions: list[dict] = []
    for key in param_keys:
        raw = [t.get(key) for t in trials]
        if any(isinstance(v, str) for v in raw if v is not None):
            unique = sorted({v for v in raw if v is not None})
            code_map = {v: i for i, v in enumerate(unique)}
            values = [code_map.get(v, -1) if v is not None else -1 for v in raw]
            dimensions.append(
                dict(
                    label=key,
                    values=values,
                    tickvals=list(range(len(unique))),
                    ticktext=unique,
                )
            )
        else:
            numeric = [float(v) if v is not None else float("nan") for v in raw]
            dimensions.append(dict(label=key, values=numeric))

    fig.add_trace(
        go.Parcoords(
            line=dict(
                color=objective_vals,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title="Objective"),
            ),
            dimensions=dimensions,
        )
    )
    fig.update_layout(title="Hyperparameter Search — Parallel Coordinates")
    return fig
