"""Unit-test fixtures: resolved paths for repo-level artefacts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Clear torch from sys.modules before the import-guard test.

    test_neural.py imports torch at collection time, which would cause torch to
    appear in sys.modules before test_root_package_does_not_import_torch_on_load
    runs.  Clearing it here lets the test verify cleanly that `import orcamind`
    does not eagerly pull in torch.
    """
    if item.name == "test_root_package_does_not_import_torch_on_load":
        for key in list(sys.modules.keys()):
            if key == "torch" or key.startswith("torch."):
                del sys.modules[key]

_REPO_ROOT = Path(__file__).parents[4]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _REPO_ROOT


@pytest.fixture
def mock_session():
    """AsyncSession mock with add (sync) and flush/execute (async) pre-wired."""
    from unittest.mock import AsyncMock, MagicMock

    from sqlalchemy.ext.asyncio import AsyncSession

    session = AsyncMock(spec=AsyncSession)
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock()
    return session


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
