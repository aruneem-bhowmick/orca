"""Pytest configuration and shared fixtures for orcalab tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[3]


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("clean_torch_modules"):
        for key in list(sys.modules.keys()):
            if key == "torch" or key.startswith("torch."):
                del sys.modules[key]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _REPO_ROOT


@pytest.fixture(scope="session")
def config_dir() -> Path:
    return _REPO_ROOT / "packages" / "orcalab" / "config"
