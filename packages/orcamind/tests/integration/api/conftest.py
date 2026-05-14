"""Shared fixtures for OrcaMind API integration tests.

All external dependencies (DB, FAISS, selectors) are mocked so no Docker
stack is required to run these tests.

Note on ASGITransport + lifespan: httpx's ASGITransport does not trigger
ASGI lifespan events, so app.state is not populated by the lifespan function.
We pre-populate app.state manually in the `client` fixture to replicate what
the lifespan would have set.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import NAMESPACE_DNS, UUID, uuid4, uuid5

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from orcamind.embedders.similarity import FaissIndex

from orca_shared.registry.repository import (
    EmbeddingRepository,
    ExperimentRepository,
    PerformanceRepository,
    TaskRepository,
)
from orca_shared.schemas.embedding import Embedding
from orca_shared.schemas.task import Task
from orca_shared.schemas.training import ExperimentResult
from orcamind.api.deps import (
    get_db,
    get_embedding_repo,
    get_experiment_repo,
    get_faiss_index,
    get_nn_selector,
    get_perf_repo,
    get_predictor,
    get_stat_embedder,
    get_task_repo,
)
from orcamind.api.main import create_app


# ---------------------------------------------------------------------------
# ID + timestamp fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def task_id() -> UUID:
    return uuid4()


@pytest.fixture
def model_id() -> UUID:
    return uuid4()


@pytest.fixture
def experiment_id() -> UUID:
    return uuid4()


@pytest.fixture
def embedding_id() -> UUID:
    return uuid4()


@pytest.fixture
def now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Mock DB session
# ---------------------------------------------------------------------------


def _make_execute_result(rows=(), scalar_one_or_none=None) -> MagicMock:
    """Build a sync mock that mimics SQLAlchemy's CursorResult for common call patterns."""
    result = MagicMock()
    result.scalars.return_value = iter(rows)
    result.scalar_one_or_none.return_value = scalar_one_or_none
    return result


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock(spec=AsyncSession)
    session.add = MagicMock()
    session.flush = AsyncMock()
    # Return a MagicMock (not AsyncMock) so that .scalars() is synchronously iterable
    session.execute = AsyncMock(return_value=_make_execute_result())
    return session


# ---------------------------------------------------------------------------
# Mock repositories
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_task_repo(task_id: UUID, now: datetime) -> AsyncMock:
    repo = AsyncMock(spec=TaskRepository)
    repo.get_by_id.return_value = Task(
        task_id=task_id,
        name="iris",
        domain="tabular",
        task_type="classification",
        created_at=now,
        updated_at=now,
    )
    repo.list_all.return_value = []
    repo.list_by_domain.return_value = []
    repo.list_by_type.return_value = []
    repo.update_embedding.return_value = None
    return repo


@pytest.fixture
def mock_experiment_repo(
    experiment_id: UUID, task_id: UUID, model_id: UUID
) -> AsyncMock:
    repo = AsyncMock(spec=ExperimentRepository)
    repo.create.return_value = ExperimentResult(
        experiment_id=experiment_id,
        task_id=task_id,
        model_id=model_id,
        status="pending",
    )
    repo.get_by_id.return_value = ExperimentResult(
        experiment_id=experiment_id,
        task_id=task_id,
        model_id=model_id,
        status="pending",
    )
    repo.update_status.return_value = None
    return repo


@pytest.fixture
def mock_embedding_repo(
    embedding_id: UUID, task_id: UUID, now: datetime
) -> AsyncMock:
    repo = AsyncMock(spec=EmbeddingRepository)
    repo.create.return_value = Embedding(
        embedding_id=embedding_id,
        task_id=task_id,
        embedding_type="statistical",
        embedding_vector=[0.0] * 25,
        dimension=25,
        model_version="v1",
        created_at=now,
    )
    return repo


@pytest.fixture
def mock_perf_repo() -> AsyncMock:
    repo = AsyncMock(spec=PerformanceRepository)
    repo.log_metric.return_value = None
    return repo


# ---------------------------------------------------------------------------
# Mock FAISS index, selectors, embedder
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_faiss_index() -> MagicMock:
    idx = MagicMock()
    idx.search.return_value = []
    return idx


@pytest.fixture
def seeded_faiss_index() -> FaissIndex:
    """Real FaissIndex with 10 deterministic UUID-keyed L2-normalised embeddings (dim=25, cosine)."""
    idx = FaissIndex(dim=25, metric="cosine")
    rng = np.random.default_rng(seed=42)
    for i in range(10):
        vec = rng.random(25).astype(np.float32)
        vec /= np.linalg.norm(vec)
        idx.add(str(uuid5(NAMESPACE_DNS, str(i))), vec)
    return idx


@pytest.fixture
def mock_nn_selector() -> MagicMock:
    sel = MagicMock()
    sel.recommend.return_value = []
    return sel


@pytest.fixture
def mock_predictor() -> MagicMock:
    pred = MagicMock()
    pred.predict_with_confidence.return_value = (0.85, 0.05)
    return pred


@pytest.fixture
def mock_stat_embedder() -> MagicMock:
    emb = MagicMock()
    emb.embed.return_value = np.zeros(25, dtype=np.float64)
    return emb


# ---------------------------------------------------------------------------
# Test client with all dependencies overridden and app.state pre-populated
# ---------------------------------------------------------------------------


def _build_app(
    mock_session,
    mock_task_repo,
    mock_experiment_repo,
    mock_embedding_repo,
    mock_perf_repo,
    faiss_index,
    mock_nn_selector,
    mock_predictor,
    mock_stat_embedder,
    *,
    state_faiss_index=None,
):
    """Wire a FastAPI app with all external dependencies replaced for testing."""
    app = create_app()

    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock()
    app.state.db_engine = mock_engine

    @asynccontextmanager
    async def _fake_sessionmaker():
        m = AsyncMock()
        m.execute.return_value = AsyncMock()
        yield m

    app.state.db_sessionmaker = _fake_sessionmaker
    app.state.faiss_index = state_faiss_index
    app.state.stat_embedder = mock_stat_embedder
    app.state.nn_selector = mock_nn_selector
    app.state.predictor = mock_predictor

    app.dependency_overrides[get_db] = lambda: mock_session
    app.dependency_overrides[get_task_repo] = lambda: mock_task_repo
    app.dependency_overrides[get_experiment_repo] = lambda: mock_experiment_repo
    app.dependency_overrides[get_embedding_repo] = lambda: mock_embedding_repo
    app.dependency_overrides[get_perf_repo] = lambda: mock_perf_repo
    app.dependency_overrides[get_faiss_index] = lambda: faiss_index
    app.dependency_overrides[get_nn_selector] = lambda: mock_nn_selector
    app.dependency_overrides[get_predictor] = lambda: mock_predictor
    app.dependency_overrides[get_stat_embedder] = lambda: mock_stat_embedder

    return app


@pytest.fixture
async def client(
    mock_session: AsyncMock,
    mock_task_repo: AsyncMock,
    mock_experiment_repo: AsyncMock,
    mock_embedding_repo: AsyncMock,
    mock_perf_repo: AsyncMock,
    mock_faiss_index: MagicMock,
    mock_nn_selector: MagicMock,
    mock_predictor: MagicMock,
    mock_stat_embedder: MagicMock,
) -> AsyncClient:
    app = _build_app(
        mock_session, mock_task_repo, mock_experiment_repo, mock_embedding_repo,
        mock_perf_repo, mock_faiss_index, mock_nn_selector, mock_predictor,
        mock_stat_embedder, state_faiss_index=None,
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def recommend_client(
    mock_session: AsyncMock,
    mock_task_repo: AsyncMock,
    mock_experiment_repo: AsyncMock,
    mock_embedding_repo: AsyncMock,
    mock_perf_repo: AsyncMock,
    seeded_faiss_index: FaissIndex,
    mock_nn_selector: MagicMock,
    mock_predictor: MagicMock,
    mock_stat_embedder: MagicMock,
) -> AsyncClient:
    """Test client that uses a real seeded FaissIndex for similar-tasks tests."""
    app = _build_app(
        mock_session, mock_task_repo, mock_experiment_repo, mock_embedding_repo,
        mock_perf_repo, seeded_faiss_index, mock_nn_selector, mock_predictor,
        mock_stat_embedder, state_faiss_index=seeded_faiss_index,
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
