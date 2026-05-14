"""Performance Heatmap – mean accuracy across tasks × model architectures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


def fetch_performances(api_url: str, metric_name: str = "accuracy") -> list[dict]:
    resp = requests.get(
        f"{api_url}/api/v1/performances",
        params={"metric_name": metric_name},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def build_heatmap_df(records: list[dict]) -> pd.DataFrame:
    """Pivot records into a (task × architecture) accuracy matrix."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df.pivot_table(
        index="task_name",
        columns="architecture",
        values="mean_accuracy",
        aggfunc="mean",
    )


# ── Streamlit page ────────────────────────────────────────────────────────────

st.title("Performance Heatmap")

api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")
metric_name = st.sidebar.text_input("Metric name", value="accuracy")

with st.spinner("Loading performance data…"):
    try:
        records = fetch_performances(api_url, metric_name=metric_name)
    except Exception as exc:
        st.error(f"Failed to fetch performance data: {exc}")
        st.stop()

if not records:
    st.info(f"No performance data found for metric '{metric_name}'.")
    st.stop()

heatmap_df = build_heatmap_df(records)

if heatmap_df.empty:
    st.warning("Could not build heatmap from the returned data.")
    st.stop()

z = heatmap_df.values.tolist()
x = heatmap_df.columns.tolist()
y = heatmap_df.index.tolist()

# Replace NaN with None so Plotly renders them in gray
z_with_none = [
    [None if (v is not None and np.isnan(v)) else v for v in row] for row in z
]

fig = go.Figure(
    data=go.Heatmap(
        z=z_with_none,
        x=x,
        y=y,
        colorscale="RdYlGn",
        zmin=0,
        zmax=1,
        colorbar={"title": "Mean accuracy"},
    )
)
fig.update_layout(
    title=f"Mean {metric_name} — tasks × architectures",
    xaxis_title="Model architecture",
    yaxis_title="Task",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Raw data")
st.dataframe(heatmap_df.style.format("{:.3f}", na_rep="—"), use_container_width=True)
