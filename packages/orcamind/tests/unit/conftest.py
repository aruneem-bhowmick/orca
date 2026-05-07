"""Unit-test fixtures: resolved paths for repo-level artefacts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[4]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _REPO_ROOT


@pytest.fixture(scope="session")
def orcamind_pkg_dir() -> Path:
    return _REPO_ROOT / "packages" / "orcamind"


@pytest.fixture(scope="session")
def config_dir() -> Path:
    return _REPO_ROOT / "packages" / "orcamind" / "config"


@pytest.fixture(scope="session")
def bmd(repo_root: Path):
    """Bootstrap script loaded as a module (without executing __main__)."""
    path = repo_root / "scripts" / "bootstrap_meta_dataset.py"
    spec = importlib.util.spec_from_file_location("bootstrap_meta_dataset", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module
