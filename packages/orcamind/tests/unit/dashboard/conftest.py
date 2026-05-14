"""Patch streamlit and its sub-modules before any dashboard page is imported.

All page modules execute top-level Streamlit calls on import; replacing
streamlit with a MagicMock lets tests import those modules and exercise
the pure data-processing functions without requiring a live Streamlit
runtime context.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="session", autouse=True)
def _patch_streamlit():
    mock_st = MagicMock()
    # Make st.stop() raise a sentinel so callers can detect it if needed.
    mock_st.stop.side_effect = SystemExit(0)

    for mod in (
        "streamlit",
        "streamlit.components",
        "streamlit.components.v1",
        "streamlit.testing",
        "streamlit.testing.v1",
    ):
        sys.modules[mod] = mock_st

    yield mock_st

    # Cleanup: remove only the keys we set (leave real streamlit alone if it
    # was already imported before this fixture ran).
    for mod in (
        "streamlit",
        "streamlit.components",
        "streamlit.components.v1",
        "streamlit.testing",
        "streamlit.testing.v1",
    ):
        sys.modules.pop(mod, None)
