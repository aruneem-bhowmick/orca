"""Training Progress – compare MLflow runs with per-epoch metric charts."""

from __future__ import annotations

import time

import mlflow
import pandas as pd
import plotly.express as px
import streamlit as st


def fetch_mlflow_runs(tracking_uri: str) -> pd.DataFrame:
    mlflow.set_tracking_uri(tracking_uri)
    try:
        return mlflow.search_runs(search_all_experiments=True)
    except Exception:
        return pd.DataFrame()


def build_metric_df(
    client: mlflow.tracking.MlflowClient,
    metric: str,
    selected_run_ids: list[str],
) -> pd.DataFrame:
    """Fetch per-epoch metric history for each selected run."""
    records: list[dict] = []
    for run_id in selected_run_ids:
        try:
            history = client.get_metric_history(run_id, metric)
            for point in history:
                records.append(
                    {"run_id": run_id, "epoch": point.step, "value": point.value}
                )
        except Exception:
            pass
    if not records:
        return pd.DataFrame(columns=["run_id", "epoch", "value"])
    return pd.DataFrame(records)


# ── Streamlit page ────────────────────────────────────────────────────────────

def _page() -> None:
    st.title("Training Progress")

    tracking_uri = st.sidebar.text_input(
        "MLflow Tracking URI", value="http://localhost:5000"
    )
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30 s)", value=False)

    with st.spinner("Loading MLflow runs…"):
        runs_df = fetch_mlflow_runs(tracking_uri)

    if runs_df.empty:
        st.info("No MLflow runs found. Check the tracking URI.")
        st.stop()

    run_options = runs_df["run_id"].tolist() if "run_id" in runs_df.columns else []
    selected_runs: list[str] = st.multiselect(
        "Select runs to compare", run_options, default=run_options[:1] if run_options else []
    )

    placeholder = st.empty()

    with placeholder.container():
        if not selected_runs:
            st.info("Select at least one run to display metrics.")
        else:
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

            for metric_key in ("meta_train_loss", "meta_train_accuracy"):
                metric_df = build_metric_df(client, metric_key, selected_runs)
                if metric_df.empty:
                    st.caption(f"No history for **{metric_key}**.")
                    continue
                fig = px.line(
                    metric_df,
                    x="epoch",
                    y="value",
                    color="run_id",
                    title=metric_key.replace("_", " ").title(),
                    labels={"value": metric_key, "epoch": "Epoch"},
                )
                st.plotly_chart(fig, use_container_width=True)

    if auto_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    _page()
