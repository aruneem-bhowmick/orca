"""Task Browser – list, filter, and inspect registered tasks with PCA projection."""

from __future__ import annotations

import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA


def fetch_tasks(
    api_url: str,
    domain: str | None = None,
    task_type: str | None = None,
) -> list[dict]:
    params: dict[str, str] = {}
    if domain:
        params["domain"] = domain
    if task_type:
        params["task_type"] = task_type
    resp = requests.get(f"{api_url}/api/v1/tasks", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_task_detail(api_url: str, task_id: str) -> dict:
    resp = requests.get(f"{api_url}/api/v1/tasks/{task_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def build_pca_df(tasks: list[dict], selected_id: str) -> pd.DataFrame | None:
    """Return a 2-column PCA DataFrame from task meta-features, or None if insufficient data."""
    features, ids, names = [], [], []
    for t in tasks:
        n_samples = t.get("n_samples") or 0
        n_features = t.get("n_features") or 0
        n_classes = t.get("n_classes") or 0
        features.append([n_samples, n_features, n_classes])
        ids.append(str(t.get("task_id", "")))
        names.append(t.get("name", ""))

    if len(features) < 2:
        return None

    n_components = min(2, len(features[0]))
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(features)

    df = pd.DataFrame(coords, columns=["PC1", "PC2"] if n_components == 2 else ["PC1"])
    if "PC2" not in df.columns:
        df["PC2"] = 0.0
    df["task_id"] = ids
    df["name"] = names
    df["selected"] = df["task_id"] == selected_id
    return df


# ── Streamlit page ────────────────────────────────────────────────────────────

st.title("Task Browser")

api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")

with st.spinner("Loading tasks…"):
    try:
        tasks = fetch_tasks(api_url)
    except Exception as exc:
        st.error(f"Failed to fetch tasks: {exc}")
        st.stop()

if not tasks:
    st.info("No tasks found.")
    st.stop()

df = pd.DataFrame(tasks)

for col in ("domain", "task_type"):
    if col not in df.columns:
        df[col] = None

col1, col2 = st.columns(2)
with col1:
    domains = ["All"] + sorted(df["domain"].dropna().unique().tolist())
    selected_domain = st.selectbox("Filter by domain", domains)
with col2:
    types = ["All"] + sorted(df["task_type"].dropna().unique().tolist())
    selected_type = st.selectbox("Filter by task type", types)

filtered = df.copy()
if selected_domain != "All":
    filtered = filtered[filtered["domain"] == selected_domain]
if selected_type != "All":
    filtered = filtered[filtered["task_type"] == selected_type]

st.dataframe(filtered, use_container_width=True)

task_ids = filtered["task_id"].astype(str).tolist() if "task_id" in filtered.columns else []
if not task_ids:
    st.stop()

selected_id = st.selectbox("Select a task to inspect", task_ids)

if selected_id:
    st.subheader("Task Details")
    try:
        detail = fetch_task_detail(api_url, selected_id)
        st.json(detail)
    except Exception as exc:
        st.warning(f"Could not load task detail: {exc}")
        detail = {}

    # 2D PCA projection of task embeddings
    numeric_tasks = [
        t for t in tasks
        if any(t.get(k) is not None for k in ("n_samples", "n_features", "n_classes"))
    ]
    pca_df = build_pca_df(numeric_tasks, selected_id) if len(numeric_tasks) >= 2 else None
    if pca_df is not None:
        st.subheader("2D PCA Projection of Task Embeddings")
        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="selected",
            color_discrete_map={True: "red", False: "steelblue"},
            hover_data=["task_id", "name"],
            title="Task Embedding Space (PCA of meta-features)",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough tasks with numerical meta-features to render PCA projection.")
