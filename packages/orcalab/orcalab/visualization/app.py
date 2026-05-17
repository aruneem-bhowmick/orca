"""OrcaLab Streamlit dashboard - main entry point and navigation."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="OrcaLab", layout="wide")

st.sidebar.text_input("OrcaLab API URL", value="http://localhost:8001")

pg = st.navigation(
    [
        st.Page("pages/live_experiments.py", title="Live Experiments"),
        st.Page("pages/search_progress.py", title="Search Progress"),
        st.Page("pages/results_explorer.py", title="Results Explorer"),
        st.Page("pages/meta_analysis.py", title="Meta-Analysis"),
    ]
)

if __name__ == "__main__":
    pg.run()
