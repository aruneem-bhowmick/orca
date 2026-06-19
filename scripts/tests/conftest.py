"""Pytest configuration and shared fixtures for scripts/ tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _REPO_ROOT


@pytest.fixture(scope="module")
def init_prefect(repo_root: Path) -> ModuleType:
    """Import ``scripts/init_prefect.py`` as a module without requiring prefect."""
    script_path = repo_root / "scripts" / "init_prefect.py"
    spec = importlib.util.spec_from_file_location("init_prefect", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("init_prefect", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bootstrap(repo_root: Path) -> ModuleType:
    """Import ``scripts/bootstrap_meta_dataset.py`` without requiring openml.

    The ``openml`` import inside ``download_suite`` and ``fetch_dataset`` is
    deferred, so the module loads cleanly as long as the heavyweight
    ``openml`` package is never installed. Tests that touch
    ``download_suite`` / ``fetch_dataset`` mock ``openml`` explicitly.
    """
    script_path = repo_root / "scripts" / "bootstrap_meta_dataset.py"
    spec = importlib.util.spec_from_file_location(
        "bootstrap_meta_dataset", script_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("bootstrap_meta_dataset", module)
    spec.loader.exec_module(module)
    return module
