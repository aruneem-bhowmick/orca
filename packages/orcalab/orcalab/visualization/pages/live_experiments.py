"""Live Experiments – auto-refreshing view of running experiments."""

from __future__ import annotations

import logging
import time

import pandas as pd
import requests
import streamlit as st

from orcalab.visualization.components.metric_plots import loss_curve

logger = logging.getLogger(__name__)

STATUS_COLORS: dict[str, str] = {
    "RUNNING": "#28a745",
    "PENDING": "#6c757d",
    "FAILED": "#dc3545",
    "COMPLETED": "#007bff",
}


def fetch_experiments(api_url: str) -> list[dict]:
    resp = requests.get(f"{api_url}/api/v1/experiments", timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_experiment_history(api_url: str, experiment_id: str) -> list[dict]:
    """Fetch per-epoch metric history for a single experiment."""
    resp = requests.get(
        f"{api_url}/api/v1/experiments/{experiment_id}/history", timeout=10
    )
    resp.raise_for_status()
    return resp.json()


def color_for_status(status: str) -> str:
    """Return a hex colour string for the given experiment status."""
    return STATUS_COLORS.get(status.upper(), "#6c757d")


def compute_progress(
    current_epoch: int | None, total_epochs: int | None
) -> float:
    """Return a 0.0–1.0 progress ratio; 0.0 for missing or invalid inputs."""
    if not current_epoch or not total_epochs or total_epochs <= 0:
        return 0.0
    return min(float(current_epoch) / float(total_epochs), 1.0)


# ── Streamlit page ────────────────────────────────────────────────────────────


def _page() -> None:
    st.title("Live Experiments")

    api_url = st.sidebar.text_input(
        "OrcaLab API URL", value="http://localhost:8001"
    )
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5 s)", value=True)

    with st.spinner("Loading experiments…"):
        try:
            experiments = fetch_experiments(api_url)
        except Exception as exc:
            st.error(f"Failed to fetch experiments: {exc}")
            st.stop()

    if not experiments:
        st.info("No experiments found.")
        if auto_refresh:
            time.sleep(5)
            st.rerun()
        st.stop()

    df = pd.DataFrame(experiments)

    status_options = (
        ["All"] + sorted(df["status"].dropna().unique().tolist())
        if "status" in df.columns
        else ["All"]
    )
    selected_status = st.sidebar.selectbox("Filter by status", status_options)
    filtered = df if selected_status == "All" else df[df["status"] == selected_status]

    st.dataframe(filtered, use_container_width=True)

    exp_ids = (
        filtered["experiment_id"].astype(str).tolist()
        if "experiment_id" in filtered.columns
        else []
    )
    if not exp_ids:
        if auto_refresh:
            time.sleep(5)
            st.rerun()
        st.stop()

    selected_id = st.selectbox("Select experiment for detail", exp_ids)

    if selected_id:
        row = filtered[filtered["experiment_id"].astype(str) == selected_id]
        if not row.empty:
            status = (
                str(row["status"].iloc[0]) if "status" in row.columns else "UNKNOWN"
            )
            color = color_for_status(status)
            st.markdown(
                f"**Status**: <span style='color:{color}'>{status}</span>",
                unsafe_allow_html=True,
            )

            current_epoch = (
                row["current_epoch"].iloc[0]
                if "current_epoch" in row.columns
                else None
            )
            total_epochs = (
                row["total_epochs"].iloc[0]
                if "total_epochs" in row.columns
                else None
            )
            progress = compute_progress(current_epoch, total_epochs)
            st.progress(
                progress,
                text=f"Epoch {current_epoch or '?'} / {total_epochs or '?'}",
            )

        with st.spinner("Loading metric history…"):
            try:
                history = fetch_experiment_history(api_url, selected_id)
                if history:
                    fig = loss_curve(history, title=f"Metrics — {selected_id[:8]}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No metric history available yet.")
            except Exception as exc:
                st.warning(f"Could not load history: {exc}")

    if auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    _page()
