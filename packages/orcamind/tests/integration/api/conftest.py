"""Shared fixtures for OrcaMind API integration tests.

All external dependencies (DB, FAISS, selectors) are mocked so no Docker
stack is required to run these tests.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from orca_shared.registry.repository import (
    EmbeddingRepository,
    ExperimentRepository,
    PerformanceRepository,
    TaskRepository,
)
from orca_shared.schemas.embedding import Embedding
from orca_shared.schemas.recommendation import ModelRecommendation
from orca_shared.schemas.task import Task, TaskSummary
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


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock(spec=AsyncSession)
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock()
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
# Test client with all dependencies overridden
# ---------------------------------------------------------------------------


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
    app = create_app()

    app.dependency_overrides[get_db] = lambda: mock_session
    app.dependency_overrides[get_task_repo] = lambda: mock_task_repo
    app.dependency_overrides[get_experiment_repo] = lambda: mock_experiment_repo
    app.dependency_overrides[get_embedding_repo] = lambda: mock_embedding_repo
    app.dependency_overrides[get_perf_repo] = lambda: mock_perf_repo
    app.dependency_overrides[get_faiss_index] = lambda: mock_faiss_index
    app.dependency_overrides[get_nn_selector] = lambda: mock_nn_selector
    app.dependency_overrides[get_predictor] = lambda: mock_predictor
    app.dependency_overrides[get_stat_embedder] = lambda: mock_stat_embedder

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
