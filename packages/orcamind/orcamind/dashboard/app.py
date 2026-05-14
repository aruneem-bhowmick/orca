"""OrcaMind Streamlit dashboard – main entry point and navigation."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="OrcaMind", layout="wide")

pg = st.navigation(
    [
        st.Page("pages/task_browser.py", title="Task Browser"),
        st.Page("pages/training_progress.py", title="Training Progress"),
        st.Page("pages/recommendation_explorer.py", title="Recommendation Explorer"),
        st.Page("pages/performance_heatmap.py", title="Performance Heatmap"),
    ]
)
if __name__ == "__main__":
    pg.run()
