"""Smoke tests for the OrcaLab dashboard app entry point."""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture(scope="module")
def app(_patch_streamlit):
    sys.modules.pop("orcalab.visualization.app", None)
    return importlib.import_module("orcalab.visualization.app")


class TestAppEntryPoint:
    def test_module_imports_without_error(self, app):
        assert app is not None

    def test_set_page_config_called(self, _patch_streamlit, app):
        _patch_streamlit.set_page_config.assert_called()

    def test_set_page_config_uses_orcalab_title(self, _patch_streamlit, app):
        calls = _patch_streamlit.set_page_config.call_args_list
        titles = [
            c.kwargs.get("page_title") or (c.args[0] if c.args else None)
            for c in calls
        ]
        assert any(t == "OrcaLab" for t in titles if t)

    def test_navigation_called(self, _patch_streamlit, app):
        _patch_streamlit.navigation.assert_called()

    def test_navigation_receives_four_pages(self, _patch_streamlit, app):
        nav_calls = _patch_streamlit.navigation.call_args_list
        assert nav_calls, "st.navigation was not called"
        pages_arg = nav_calls[0].args[0] if nav_calls[0].args else nav_calls[0].kwargs.get("pages", [])
        assert len(pages_arg) == 4

    def test_sidebar_api_url_input_declared(self, _patch_streamlit, app):
        sidebar = _patch_streamlit.sidebar
        inputs = sidebar.text_input.call_args_list
        labels = [
            c.args[0] if c.args else c.kwargs.get("label", "")
            for c in inputs
        ]
        assert any("OrcaLab API URL" in str(label) for label in labels)
