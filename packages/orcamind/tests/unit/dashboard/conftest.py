"""Patch UI/visualisation modules before any dashboard page is imported.

All page modules execute top-level Streamlit and Plotly calls on import;
replacing them with MagicMocks lets tests import those modules and exercise
the pure data-processing functions without requiring a live Streamlit
runtime context or plotly to be installed.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

_MOCKED_MODULES = (
    "streamlit",
    "streamlit.components",
    "streamlit.components.v1",
    "streamlit.testing",
    "streamlit.testing.v1",
    "plotly",
    "plotly.express",
    "plotly.graph_objects",
)


@pytest.fixture(scope="session", autouse=True)
def _patch_streamlit():
    mock_st = MagicMock()

    for mod in _MOCKED_MODULES:
        sys.modules[mod] = MagicMock()

    # Return the streamlit mock so tests can assert on st.* calls.
    sys.modules["streamlit"] = mock_st

    yield mock_st

    # Cleanup: remove only the keys we set.
    for mod in _MOCKED_MODULES:
        sys.modules.pop(mod, None)
