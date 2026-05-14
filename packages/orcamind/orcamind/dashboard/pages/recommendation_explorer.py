"""Recommendation Explorer – upload a CSV dataset and get ranked model recommendations."""

from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


def parse_csv_dataset(uploaded_file: io.BytesIO) -> dict:
    """Derive basic dataset statistics from an uploaded CSV file."""
    df = pd.read_csv(uploaded_file)
    n_samples, total_cols = df.shape
    n_features = max(total_cols - 1, 1)
    n_classes = int(df.iloc[:, -1].nunique())
    return {
        "name": getattr(uploaded_file, "name", "uploaded_dataset"),
        "task_type": "classification",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
    }


def call_embed(api_url: str, dataset_summary: dict) -> list[float]:
    resp = requests.post(
        f"{api_url}/api/v1/tasks/embed", json=dataset_summary, timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    vector = data.get("embedding_vector") or data.get("embedding") or data
    if isinstance(vector, list):
        return vector
    raise ValueError(f"Unexpected embed response: {data}")


def call_recommend(
    api_url: str, embedding: list[float], top_k: int = 3
) -> list[dict]:
    payload = {"task_embedding": embedding, "top_k": top_k}
    resp = requests.post(
        f"{api_url}/api/v1/recommend-model", json=payload, timeout=30
    )
    resp.raise_for_status()
    return resp.json()


def call_similar_tasks(
    api_url: str, embedding: list[float], top_k: int = 5
) -> list[dict]:
    payload = {"task_embedding": embedding, "top_k": top_k}
    resp = requests.post(
        f"{api_url}/api/v1/similar-tasks", json=payload, timeout=30
    )
    resp.raise_for_status()
    return resp.json()


# ── Streamlit page ────────────────────────────────────────────────────────────

st.title("Recommendation Explorer")

api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")

uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV file to receive model recommendations.")
    st.stop()

with st.spinner("Parsing dataset…"):
    try:
        dataset_summary = parse_csv_dataset(uploaded)
    except Exception as exc:
        st.error(f"Failed to parse CSV: {exc}")
        st.stop()

st.write(
    f"Detected **{dataset_summary['n_samples']}** samples, "
    f"**{dataset_summary['n_features']}** features, "
    f"**{dataset_summary['n_classes']}** classes."
)

with st.spinner("Computing task embedding…"):
    try:
        embedding = call_embed(api_url, dataset_summary)
    except Exception as exc:
        st.error(f"Embed API call failed: {exc}")
        st.stop()

with st.spinner("Fetching recommendations…"):
    try:
        recommendations = call_recommend(api_url, embedding, top_k=3)
    except Exception as exc:
        st.error(f"Recommendation API call failed: {exc}")
        recommendations = []

# Top-3 recommendation cards
if recommendations:
    st.subheader("Top-3 Model Recommendations")
    cols = st.columns(min(3, len(recommendations)))
    for col, rec in zip(cols, recommendations[:3]):
        with col:
            score = rec.get("predicted_score", 0.0)
            arch = rec.get("architecture") or str(rec.get("model_id", "Unknown"))
            st.metric(label=arch, value=f"{score:.3f}", delta=None)
            st.json(rec)

# Similar tasks bar chart
with st.spinner("Fetching similar tasks…"):
    try:
        similar = call_similar_tasks(api_url, embedding, top_k=5)
    except Exception as exc:
        st.warning(f"Similar-tasks API call failed: {exc}")
        similar = []

if similar:
    st.subheader("Similar Tasks")
    sim_df = pd.DataFrame(similar)
    score_col = next(
        (c for c in ("score", "similarity", "similarity_score") if c in sim_df.columns),
        sim_df.columns[-1] if len(sim_df.columns) > 0 else None,
    )
    name_col = next(
        (c for c in ("name", "task_name", "task_id") if c in sim_df.columns),
        sim_df.columns[0] if len(sim_df.columns) > 0 else None,
    )
    if score_col and name_col:
        fig = px.bar(
            sim_df,
            x=name_col,
            y=score_col,
            title="Similarity Scores",
            labels={name_col: "Task", score_col: "Similarity"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(sim_df, use_container_width=True)
