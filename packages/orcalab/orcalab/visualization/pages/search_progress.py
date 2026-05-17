"""Search Progress – parallel coordinates view of hyperparameter sweeps."""

from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from orcalab.visualization.components.parallel_coords import parallel_coordinates

logger = logging.getLogger(__name__)


def fetch_sweeps(api_url: str) -> list[dict]:
    resp = requests.get(f"{api_url}/api/v1/sweeps", timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_sweep_trials(api_url: str, sweep_id: str) -> list[dict]:
    """Fetch all trials for a given sweep."""
    resp = requests.get(f"{api_url}/api/v1/sweeps/{sweep_id}/trials", timeout=10)
    resp.raise_for_status()
    return resp.json()


def find_best_trial(trials: list[dict]) -> dict | None:
    """Return the trial with the highest objective; ``None`` if trials is empty."""
    valid = [t for t in trials if t.get("objective") is not None]
    if not valid:
        return None
    return max(valid, key=lambda t: t["objective"])


def build_cumulative_df(trials: list[dict]) -> pd.DataFrame:
    """Build a DataFrame with columns ``[trial_index, cumulative_count]``."""
    if not trials:
        return pd.DataFrame(columns=["trial_index", "cumulative_count"])
    n = len(trials)
    return pd.DataFrame(
        {"trial_index": list(range(1, n + 1)), "cumulative_count": list(range(1, n + 1))}
    )


# ── Streamlit page ────────────────────────────────────────────────────────────


def _page() -> None:
    st.title("Search Progress")

    api_url = st.sidebar.text_input(
        "OrcaLab API URL", value="http://localhost:8001"
    )

    with st.spinner("Loading sweeps…"):
        try:
            sweeps = fetch_sweeps(api_url)
        except Exception as exc:
            st.error(f"Failed to fetch sweeps: {exc}")
            st.stop()

    if not sweeps:
        st.info("No sweeps found.")
        st.stop()

    sweep_options = {str(s.get("sweep_id", i)): s for i, s in enumerate(sweeps)}
    selected_sweep_id = st.selectbox("Select sweep", list(sweep_options.keys()))

    with st.spinner("Loading trials…"):
        try:
            trials = fetch_sweep_trials(api_url, selected_sweep_id)
        except Exception as exc:
            st.error(f"Failed to fetch trials: {exc}")
            st.stop()

    best = find_best_trial(trials)
    if best:
        st.sidebar.subheader("Best Trial")
        st.sidebar.metric("Best Objective", f"{best.get('objective', 0.0):.4f}")
        st.sidebar.json({k: v for k, v in best.items() if k != "objective"})

    if not trials:
        st.info("No trials found for this sweep.")
        st.stop()

    st.subheader("Parallel Coordinates")
    fig = parallel_coordinates(trials)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cumulative Trial Count")
    cum_df = build_cumulative_df(trials)
    cum_fig = px.line(
        cum_df,
        x="trial_index",
        y="cumulative_count",
        title="Trials over time",
        labels={"trial_index": "Trial #", "cumulative_count": "Cumulative count"},
    )
    st.plotly_chart(cum_fig, use_container_width=True)


if __name__ == "__main__":
    _page()
