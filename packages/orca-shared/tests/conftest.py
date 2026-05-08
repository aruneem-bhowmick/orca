from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession


# ---------------------------------------------------------------------------
# Storage fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def storage_base_path(tmp_path: Path) -> Path:
    """Temporary directory for local storage backend tests."""
    base = tmp_path / "orca_storage"
    base.mkdir()
    return base


# ---------------------------------------------------------------------------
# Common ID / timestamp fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture
def task_id():
    return uuid4()


@pytest.fixture
def model_id():
    return uuid4()


@pytest.fixture
def experiment_id():
    return uuid4()


@pytest.fixture
def embedding_id():
    return uuid4()


# ---------------------------------------------------------------------------
# Database / session fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_session() -> AsyncMock:
    """AsyncSession mock with add (sync) and flush/execute (async) pre-wired."""
    session = AsyncMock(spec=AsyncSession)
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock()
    return session


def make_execute_result(rows: list) -> MagicMock:
    """Return a mock CursorResult backed by *rows*."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = rows[0] if rows else None
    result.scalars.side_effect = lambda: iter(rows)
    return result


# ---------------------------------------------------------------------------
# ORM row helpers
# ---------------------------------------------------------------------------


def make_task_row(**kwargs) -> SimpleNamespace:
    """Minimal TaskORM-shaped namespace for repository tests."""
    defaults = dict(
        task_id=uuid4(),
        name="iris",
        domain="tabular",
        task_type="classification",
        n_samples=150,
        n_features=4,
        n_classes=3,
        dataset_uri=None,
        task_metadata=None,
        embedding_id=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def make_experiment_row(**kwargs) -> SimpleNamespace:
    """Minimal ExperimentORM-shaped namespace for repository tests."""
    defaults = dict(
        experiment_id=uuid4(),
        task_id=uuid4(),
        model_id=uuid4(),
        training_config=None,
        status="pending",
        mlflow_run_id=None,
        started_at=None,
        completed_at=None,
        created_by=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def make_performance_row(**kwargs) -> SimpleNamespace:
    """Minimal PerformanceORM-shaped namespace for repository tests."""
    defaults = dict(
        performance_id=uuid4(),
        experiment_id=uuid4(),
        metric_name="accuracy",
        metric_value=0.95,
        epoch=1,
        is_final=False,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# MLflow / torch sys.modules mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mlflow(monkeypatch) -> MagicMock:
    """Inject a MagicMock as the mlflow module for the duration of the test."""
    m = MagicMock()
    monkeypatch.setitem(sys.modules, "mlflow", m)
    return m


@pytest.fixture
def mock_torch(monkeypatch) -> MagicMock:
    """Inject a MagicMock as the torch module for the duration of the test."""
    t = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", t)
    return t
