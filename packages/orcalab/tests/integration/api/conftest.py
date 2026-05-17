"""Shared fixtures for OrcaLab API integration tests.

All external dependencies (DB, Prefect) are mocked so no Docker stack is
required. ASGITransport does not trigger ASGI lifespan events, so app.state
is pre-populated manually in the ``client`` fixture.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from orca_shared.registry.repository import ExperimentRepository, SearchSpaceRepository
from orca_shared.schemas.search_space import SearchSpaceRecord
from orca_shared.schemas.training import ExperimentResult
from orcalab.api.deps import get_db, get_experiment_repo, get_search_space_repo, get_sweeps_store
from orcalab.api.main import create_app


# ---------------------------------------------------------------------------
# ID + timestamp fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def experiment_id() -> UUID:
    return uuid4()


@pytest.fixture
def search_space_id() -> UUID:
    return uuid4()


@pytest.fixture
def task_id() -> UUID:
    return uuid4()


@pytest.fixture
def now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Mock session
# ---------------------------------------------------------------------------


def _make_execute_result(rows=(), scalar_one_or_none=None) -> MagicMock:
    result = MagicMock()
    result.scalars.return_value = iter(rows)
    result.scalar_one_or_none.return_value = scalar_one_or_none
    return result


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock(spec=AsyncSession)
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock(return_value=_make_execute_result())
    return session


# ---------------------------------------------------------------------------
# Mock repositories
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_experiment_repo(experiment_id: UUID, task_id: UUID, now: datetime) -> AsyncMock:
    repo = AsyncMock(spec=ExperimentRepository)
    repo.create.return_value = ExperimentResult(
        experiment_id=experiment_id,
        task_id=task_id,
        status="pending",
        started_at=now,
    )
    repo.get_by_id.return_value = ExperimentResult(
        experiment_id=experiment_id,
        task_id=task_id,
        status="pending",
        started_at=now,
    )
    repo.list_all.return_value = []
    repo.update_status.return_value = None
    repo.mark_complete.return_value = None
    return repo


@pytest.fixture
def mock_search_space_repo(search_space_id: UUID, now: datetime) -> AsyncMock:
    repo = AsyncMock(spec=SearchSpaceRepository)
    repo.create.return_value = SearchSpaceRecord(
        search_space_id=search_space_id,
        name="test_space",
        definition={"name": "test_space", "description": "", "parameters": []},
        created_at=now,
    )
    repo.list_all.return_value = []
    return repo


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------


def _build_app(
    mock_session: AsyncMock,
    mock_experiment_repo: AsyncMock,
    mock_search_space_repo: AsyncMock,
    sweeps_store: dict,
) -> object:
    app = create_app()

    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock()
    app.state.db_engine = mock_engine

    @asynccontextmanager
    async def _fake_sessionmaker():
        m = AsyncMock()
        m.__aenter__ = AsyncMock(return_value=m)
        m.__aexit__ = AsyncMock(return_value=False)
        m.execute = AsyncMock(return_value=MagicMock())
        yield m

    app.state.db_sessionmaker = _fake_sessionmaker
    app.state.sweeps = sweeps_store

    app.dependency_overrides[get_db] = lambda: mock_session
    app.dependency_overrides[get_experiment_repo] = lambda: mock_experiment_repo
    app.dependency_overrides[get_search_space_repo] = lambda: mock_search_space_repo
    app.dependency_overrides[get_sweeps_store] = lambda: sweeps_store

    return app


@pytest.fixture
def sweeps_store() -> dict:
    return {}


@pytest.fixture
async def client(
    mock_session: AsyncMock,
    mock_experiment_repo: AsyncMock,
    mock_search_space_repo: AsyncMock,
    sweeps_store: dict,
) -> AsyncClient:
    app = _build_app(mock_session, mock_experiment_repo, mock_search_space_repo, sweeps_store)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
