"""Results Explorer – completed experiments table and side-by-side comparison."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import requests
import streamlit as st

from orcalab.visualization.components.metric_plots import metric_comparison

logger = logging.getLogger(__name__)


def fetch_completed_experiments(api_url: str) -> list[dict]:
    resp = requests.get(
        f"{api_url}/api/v1/experiments",
        params={"status": "COMPLETED"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def filter_experiments(
    experiments: list[dict],
    *,
    task_id: str | None = None,
    domain: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> list[dict]:
    """Apply optional task_id, domain, and date-range filters.

    Experiments that lack the filtered field are excluded when a filter is active.
    Dates with unrecognisable ``completed_at`` strings are excluded.
    """
    result = []
    for exp in experiments:
        if task_id and str(exp.get("task_id", "")) != task_id:
            continue
        if domain and str(exp.get("domain", "")) != domain:
            continue
        if date_from is not None or date_to is not None:
            raw = exp.get("completed_at")
            if not raw:
                continue
            try:
                completed_date = date.fromisoformat(str(raw)[:10])
            except ValueError:
                continue
            if date_from is not None and completed_date < date_from:
                continue
            if date_to is not None and completed_date > date_to:
                continue
        result.append(exp)
    return result


def diff_configs(exp_a: dict, exp_b: dict) -> dict:
    """Return a dict of keys where the two experiment dicts differ."""
    all_keys = set(exp_a) | set(exp_b)
    return {
        k: {"a": exp_a.get(k), "b": exp_b.get(k)}
        for k in sorted(all_keys)
        if exp_a.get(k) != exp_b.get(k)
    }


# ── Streamlit page ────────────────────────────────────────────────────────────


def _page() -> None:
    st.title("Results Explorer")

    api_url = st.sidebar.text_input(
        "OrcaLab API URL", value="http://localhost:8001"
    )

    with st.spinner("Loading completed experiments…"):
        try:
            experiments = fetch_completed_experiments(api_url)
        except Exception as exc:
            st.error(f"Failed to fetch experiments: {exc}")
            st.stop()

    if not experiments:
        st.info("No completed experiments found.")
        st.stop()

    df = pd.DataFrame(experiments)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        task_ids = (
            ["All"] + sorted(df["task_id"].dropna().astype(str).unique().tolist())
            if "task_id" in df.columns
            else ["All"]
        )
        selected_task = st.selectbox("Filter by task_id", task_ids)
    with col2:
        domains = (
            ["All"] + sorted(df["domain"].dropna().unique().tolist())
            if "domain" in df.columns
            else ["All"]
        )
        selected_domain = st.selectbox("Filter by domain", domains)
    with col3:
        date_from = st.date_input("From date", value=None)
    with col4:
        date_to = st.date_input("To date", value=None)

    filtered_exps = filter_experiments(
        experiments,
        task_id=None if selected_task == "All" else selected_task,
        domain=None if selected_domain == "All" else selected_domain,
        date_from=date_from or None,
        date_to=date_to or None,
    )

    filtered_df = pd.DataFrame(filtered_exps) if filtered_exps else pd.DataFrame()
    st.dataframe(filtered_df, use_container_width=True)

    if len(filtered_exps) < 2:
        st.info("Select at least two experiments in the table above to compare.")
        st.stop()

    exp_ids = [str(e.get("experiment_id", i)) for i, e in enumerate(filtered_exps)]
    exp_map = {str(e.get("experiment_id", i)): e for i, e in enumerate(filtered_exps)}

    col_a, col_b = st.columns(2)
    with col_a:
        id_a = st.selectbox("Experiment A", exp_ids, key="cmp_a")
    with col_b:
        remaining = [x for x in exp_ids if x != id_a]
        id_b = st.selectbox("Experiment B", remaining, key="cmp_b")

    if id_a and id_b and id_a in exp_map and id_b in exp_map:
        diff = diff_configs(exp_map[id_a], exp_map[id_b])
        st.subheader("Config diff")
        if diff:
            st.json(diff)
        else:
            st.success("Configurations are identical.")

        from orca_shared.schemas.training import ExperimentResult

        exp_objs = []
        for eid in (id_a, id_b):
            try:
                exp_objs.append(ExperimentResult(**exp_map[eid]))
            except Exception as exc:
                logger.warning("Failed to parse ExperimentResult for %s: %s", eid, exc)

        if exp_objs:
            all_metrics: set[str] = set()
            for r in exp_objs:
                if r.metrics:
                    all_metrics.update(r.metrics.keys())
            if all_metrics:
                chosen_metric = st.selectbox(
                    "Compare metric", sorted(all_metrics)
                )
                cmp_fig = metric_comparison(exp_objs, chosen_metric)
                st.plotly_chart(cmp_fig, use_container_width=True)


if __name__ == "__main__":
    _page()
