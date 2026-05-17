"""Meta-Analysis – heatmap, scatter, and trend charts across all experiments."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

logger = logging.getLogger(__name__)


def fetch_all_experiments(api_url: str) -> list[dict]:
    resp = requests.get(f"{api_url}/api/v1/experiments", timeout=10)
    resp.raise_for_status()
    return resp.json()


def build_domain_arch_heatmap(
    experiments: list[dict], metric: str = "accuracy"
) -> pd.DataFrame:
    """Pivot table: domain (rows) × architecture (columns) → mean metric.

    Experiments missing domain, architecture, or the requested metric are
    excluded.  Returns an empty ``DataFrame`` when no valid records exist.
    """
    records = []
    for e in experiments:
        domain = e.get("domain")
        arch = e.get("architecture")
        if arch is None and isinstance(e.get("arch_config"), dict):
            arch = e["arch_config"].get("name")
        val = (e.get("metrics") or {}).get(metric)
        if domain and arch and val is not None:
            records.append({"domain": domain, "architecture": arch, "value": float(val)})
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df.pivot_table(
        index="domain", columns="architecture", values="value", aggfunc="mean"
    )


def build_scatter_df(
    experiments: list[dict], metric: str = "accuracy"
) -> pd.DataFrame:
    """Build a DataFrame for complexity-vs-accuracy scatter.

    Complexity proxy: ``n_features × n_samples`` from experiment metadata.
    Experiments lacking both n_features and n_samples are excluded.
    """
    records = []
    for e in experiments:
        n_f = e.get("n_features")
        n_s = e.get("n_samples")
        val = (e.get("metrics") or {}).get(metric)
        if n_f is None or n_s is None or val is None:
            continue
        try:
            complexity = int(n_f) * int(n_s)
        except (TypeError, ValueError):
            continue
        records.append(
            {
                "complexity": complexity,
                "accuracy": float(val),
                "experiment_id": str(e.get("experiment_id", ""))[:8],
            }
        )
    if not records:
        return pd.DataFrame(columns=["complexity", "accuracy", "experiment_id"])
    return pd.DataFrame(records)


def build_trend_df(
    experiments: list[dict], metric: str = "accuracy"
) -> pd.DataFrame:
    """Time-series of cumulative best metric value.

    Experiments without a ``completed_at`` timestamp or without the metric are
    excluded.  Returns columns ``[completed_at, value, best_so_far]``.
    """
    records = []
    for e in experiments:
        completed_at = e.get("completed_at")
        val = (e.get("metrics") or {}).get(metric)
        if completed_at is None or val is None:
            continue
        ts = pd.to_datetime(completed_at, errors="coerce")
        if pd.isna(ts):
            continue
        records.append({"completed_at": ts, "value": float(val)})
    if not records:
        return pd.DataFrame(columns=["completed_at", "value", "best_so_far"])
    df = pd.DataFrame(records).sort_values("completed_at").reset_index(drop=True)
    df["best_so_far"] = df["value"].cummax()
    return df


# ── Streamlit page ────────────────────────────────────────────────────────────


def _page() -> None:
    st.title("Meta-Analysis")

    api_url = st.sidebar.text_input(
        "OrcaLab API URL", value="http://localhost:8001"
    )
    metric = st.sidebar.text_input("Metric", value="accuracy")

    with st.spinner("Loading experiments…"):
        try:
            experiments = fetch_all_experiments(api_url)
        except Exception as exc:
            st.error(f"Failed to fetch experiments: {exc}")
            st.stop()

    if not experiments:
        st.info("No experiments found.")
        st.stop()

    # ── Domain × Architecture heatmap ────────────────────────────────────────
    st.subheader("Domain × Architecture Heatmap")
    heatmap_df = build_domain_arch_heatmap(experiments, metric=metric)
    if heatmap_df.empty:
        st.info(
            "Insufficient data for heatmap (need domain, architecture, and metric fields)."
        )
    else:
        z_raw = heatmap_df.values.tolist()
        z_safe = [
            [None if (v is not None and np.isnan(v)) else v for v in row]
            for row in z_raw
        ]
        heat_fig = go.Figure(
            data=go.Heatmap(
                z=z_safe,
                x=heatmap_df.columns.tolist(),
                y=heatmap_df.index.tolist(),
                colorscale="RdYlGn",
                colorbar={"title": f"Mean {metric}"},
            )
        )
        heat_fig.update_layout(
            xaxis_title="Architecture",
            yaxis_title="Domain",
            title=f"Mean {metric}: domains × architectures",
        )
        st.plotly_chart(heat_fig, use_container_width=True)

    # ── Complexity vs accuracy scatter ────────────────────────────────────────
    st.subheader("Task Complexity vs Best Accuracy")
    scatter_df = build_scatter_df(experiments, metric=metric)
    if scatter_df.empty:
        st.info(
            "No task complexity data available "
            "(need n_features and n_samples in experiment records)."
        )
    else:
        scatter_fig = px.scatter(
            scatter_df,
            x="complexity",
            y="accuracy",
            hover_data=["experiment_id"],
            title="Task complexity (n_features × n_samples) vs accuracy",
            labels={"complexity": "Complexity proxy", "accuracy": metric},
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    # ── Improvement over time ─────────────────────────────────────────────────
    st.subheader("Improvement Over Time")
    trend_df = build_trend_df(experiments, metric=metric)
    if trend_df.empty:
        st.info("No timestamped experiment data available.")
    else:
        trend_fig = px.line(
            trend_df,
            x="completed_at",
            y="best_so_far",
            title=f"Best {metric} seen so far over time",
            labels={"completed_at": "Completed at", "best_so_far": f"Best {metric}"},
        )
        st.plotly_chart(trend_fig, use_container_width=True)


if __name__ == "__main__":
    _page()
