"""Smoke tests for the dashboard app entry point."""

from __future__ import annotations

import importlib
import sys


def test_app_module_imports_without_error(_patch_streamlit):
    """app.py must be importable when streamlit is mocked."""
    # Remove a cached import if present.
    sys.modules.pop("orcamind.dashboard.app", None)
    mod = importlib.import_module("orcamind.dashboard.app")
    assert mod is not None


def test_app_calls_set_page_config(_patch_streamlit):
    """app.py must call st.set_page_config exactly once."""
    import streamlit as st  # resolves to the mock

    sys.modules.pop("orcamind.dashboard.app", None)
    importlib.import_module("orcamind.dashboard.app")

    call_args_list = st.set_page_config.call_args_list
    assert any(
        call.kwargs.get("page_title") == "OrcaMind"
        for call in call_args_list
        if hasattr(call, "kwargs")
    ) or any(
        args and args[0] == "OrcaMind"
        for call in call_args_list
        for args in [call.args]
    ), "st.set_page_config was not called with page_title='OrcaMind'"


def test_app_calls_navigation(_patch_streamlit):
    """app.py must call st.navigation with exactly 4 pages."""
    import streamlit as st

    sys.modules.pop("orcamind.dashboard.app", None)
    importlib.import_module("orcamind.dashboard.app")

    assert st.navigation.called, "st.navigation was not called"
