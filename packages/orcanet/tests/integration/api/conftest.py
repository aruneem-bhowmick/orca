"""Shared fixtures for OrcaNet API integration tests.

All external dependencies (DB, LLM, OrcaMind, OrcaLab) are mocked so no
Docker stack is required.  ASGITransport does not trigger ASGI lifespan
events, so app.state is pre-populated manually in the ``client`` fixture.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
import torch
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from orca_shared.registry.repository import TaskRepository
from orca_shared.schemas.embedding import SimilarityResult
from orca_shared.schemas.task import Task
from orca_shared.schemas.transfer import TransferMapping
from orcanet.api.deps import (
    get_cross_domain_embedder,
    get_db,
    get_hybrid_retriever,
    get_orcanet_agent,
    get_orcamind_client,
    get_orcalab_client,
    get_task_repo,
    get_transfer_strategies,
)
from orcanet.api.main import create_app
from orcanet.reasoning.validators import (
    SourceTaskRecommendation,
    TransferRecommendationResponse,
)
from orcanet.transfer.types import TransferScore


# ---------------------------------------------------------------------------
# ID and timestamp fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def source_task_id() -> UUID:
    return uuid4()


@pytest.fixture
def target_task_id() -> UUID:
    return uuid4()


@pytest.fixture
def mapping_id() -> UUID:
    return uuid4()


@pytest.fixture
def now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Canonical task fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def source_task(source_task_id: UUID, now: datetime) -> Task:
    return Task(
        task_id=source_task_id,
        name="source-task",
        domain="vision",
        task_type="classification",
        n_samples=1000,
        n_features=25,
        n_classes=10,
        dataset_uri=None,
        metadata=None,
        embedding_id=None,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def target_task(target_task_id: UUID, now: datetime) -> Task:
    return Task(
        task_id=target_task_id,
        name="target-task",
        domain="nlp",
        task_type="classification",
        n_samples=500,
        n_features=25,
        n_classes=5,
        dataset_uri=None,
        metadata=None,
        embedding_id=None,
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# Canned LLM agent response
# ---------------------------------------------------------------------------


@pytest.fixture
def canned_recommendation(source_task_id: UUID) -> TransferRecommendationResponse:
    return TransferRecommendationResponse(
        top_sources=[
            SourceTaskRecommendation(
                task_id=str(source_task_id),
                task_name="source-task",
                similarity_score=0.85,
                transfer_score=0.75,
                reasoning="High feature overlap",
            )
        ],
        recommended_strategy="feature",
        expected_improvement=0.12,
        explanation="Feature transfer recommended due to similar domain structure.",
        confidence=0.8,
    )


# ---------------------------------------------------------------------------
# Mock DB session
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
# Mock repositories and services
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_task_repo(source_task: Task, target_task: Task) -> AsyncMock:
    repo = AsyncMock(spec=TaskRepository)

    async def _get_by_id(task_id: UUID):
        if task_id == source_task.task_id:
            return source_task
        if task_id == target_task.task_id:
            return target_task
        return None

    repo.get_by_id.side_effect = _get_by_id
    return repo


@pytest.fixture
def mock_agent(canned_recommendation: TransferRecommendationResponse) -> AsyncMock:
    agent = AsyncMock()
    agent.recommend_transfer = AsyncMock(return_value=canned_recommendation)
    agent.llm = AsyncMock()
    return agent


@pytest.fixture
def mock_retriever(source_task: Task) -> AsyncMock:
    retriever = AsyncMock()
    retriever.retrieve = AsyncMock(
        return_value=[(source_task, 0.9, "vector similarity")]
    )
    retriever.retrieve_with_expanded_queries = AsyncMock(
        return_value=[(source_task, 0.9, "vector similarity")]
    )
    return retriever


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed = MagicMock(return_value=torch.zeros(1, 64))
    return embedder


@pytest.fixture
def mock_transfer_score() -> TransferScore:
    return TransferScore(
        overall=0.72,
        layer_scores={"layer1": 0.8, "layer2": 0.64},
        recommended_layers=["layer1"],
        reasoning="Good CKA alignment",
    )


@pytest.fixture
def mock_transfer_strategies(mock_transfer_score: TransferScore) -> dict:
    strategy = MagicMock()
    strategy.score_transfer = MagicMock(return_value=mock_transfer_score)
    return {"feature": strategy, "weight": strategy, "architecture": strategy}


@pytest.fixture
def mock_orcamind_client() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_orcalab_client() -> AsyncMock:
    return AsyncMock()


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------


def _build_app(
    mock_session: AsyncMock,
    mock_task_repo: AsyncMock,
    mock_agent: AsyncMock,
    mock_retriever: AsyncMock,
    mock_embedder: MagicMock,
    mock_transfer_strategies: dict,
    mock_orcamind_client: AsyncMock,
    mock_orcalab_client: AsyncMock,
    orcamind_url: str = "http://orcamind-test",
    orcalab_url: str = "http://orcalab-test",
) -> object:
    app = create_app()

    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock()
    app.state.db_engine = mock_engine

    @asynccontextmanager
    async def _fake_sessionmaker():
        m = AsyncMock(spec=AsyncSession)
        m.__aenter__ = AsyncMock(return_value=m)
        m.__aexit__ = AsyncMock(return_value=False)
        m.execute = AsyncMock(return_value=MagicMock())
        yield m

    app.state.db_sessionmaker = _fake_sessionmaker
    app.state.agent = mock_agent
    app.state.retriever = mock_retriever
    app.state.embedder = mock_embedder
    app.state.transfer_strategies = mock_transfer_strategies
    app.state.orcamind_client = mock_orcamind_client
    app.state.orcalab_client = mock_orcalab_client
    app.state.orcamind_url = orcamind_url
    app.state.orcalab_url = orcalab_url

    app.dependency_overrides[get_db] = lambda: mock_session
    app.dependency_overrides[get_task_repo] = lambda: mock_task_repo
    app.dependency_overrides[get_orcanet_agent] = lambda: mock_agent
    app.dependency_overrides[get_hybrid_retriever] = lambda: mock_retriever
    app.dependency_overrides[get_cross_domain_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_transfer_strategies] = lambda: mock_transfer_strategies
    app.dependency_overrides[get_orcamind_client] = lambda: mock_orcamind_client
    app.dependency_overrides[get_orcalab_client] = lambda: mock_orcalab_client

    return app


@pytest.fixture
def app(
    mock_session: AsyncMock,
    mock_task_repo: AsyncMock,
    mock_agent: AsyncMock,
    mock_retriever: AsyncMock,
    mock_embedder: MagicMock,
    mock_transfer_strategies: dict,
    mock_orcamind_client: AsyncMock,
    mock_orcalab_client: AsyncMock,
):
    return _build_app(
        mock_session,
        mock_task_repo,
        mock_agent,
        mock_retriever,
        mock_embedder,
        mock_transfer_strategies,
        mock_orcamind_client,
        mock_orcalab_client,
    )


@pytest_asyncio.fixture
async def client(app) -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
