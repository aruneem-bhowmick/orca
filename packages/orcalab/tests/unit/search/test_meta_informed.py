"""Unit tests for MetaInformedSearch."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from orca_shared.clients.orcamind_client import OrcaMindClient
from orca_shared.schemas.embedding import Embedding, SimilarityResult
from orca_shared.schemas.metrics import PerformanceMetrics
from orca_shared.schemas.recommendation import FeedbackRequest, ModelRecommendation
from orcalab.search.bayesian import BayesianSearch
from orcalab.search.meta_informed import MetaInformedSearch
from orcalab.search_spaces.parameters import FloatParameter, IntParameter
from orcalab.search_spaces.space import SearchSpace

# ---------------------------------------------------------------------------
# Shared identifiers
# ---------------------------------------------------------------------------

_TASK_UUID = uuid.uuid4()
_TASK_ID = str(_TASK_UUID)
_MODEL_UUID = uuid.uuid4()


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _embedding(dim: int = 4) -> Embedding:
    return Embedding(
        embedding_id=uuid.uuid4(),
        task_id=_TASK_UUID,
        embedding_vector=[0.1] * dim,
        dimension=dim,
        created_at=datetime.now(timezone.utc),
    )


def _recommendation(predicted_score: float = 0.85) -> ModelRecommendation:
    return ModelRecommendation(
        task_id=_TASK_UUID,
        model_id=_MODEL_UUID,
        predicted_score=predicted_score,
    )


def _similar_tasks(n: int = 3) -> list[SimilarityResult]:
    return [
        SimilarityResult(task_id=uuid.uuid4(), score=0.9 - i * 0.1, rank=i + 1)
        for i in range(n)
    ]


def _perf_metrics(accuracy: float = 0.80) -> PerformanceMetrics:
    return PerformanceMetrics(
        experiment_id=uuid.uuid4(),
        final_metrics={"accuracy": accuracy},
    )


def _perf_metrics_no_accuracy() -> PerformanceMetrics:
    return PerformanceMetrics(
        experiment_id=uuid.uuid4(),
        final_metrics={"loss": 0.3},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock(spec=OrcaMindClient)
    client.embed_task = AsyncMock(return_value=_embedding())
    client.recommend_model = AsyncMock(return_value=_recommendation())
    client.find_similar_tasks = AsyncMock(return_value=_similar_tasks(3))
    client.predict_performance = AsyncMock(return_value=_perf_metrics())
    client.submit_feedback = AsyncMock(return_value=None)
    return client


@pytest.fixture
def simple_space() -> SearchSpace:
    return (
        SearchSpace("meta_test")
        .add(FloatParameter("lr", low=1e-4, high=1e-1))
        .add(IntParameter("layers", low=2, high=8))
    )


# ---------------------------------------------------------------------------
# TestInitializeFromOrcamind
# ---------------------------------------------------------------------------


class TestInitializeFromOrcamind:
    async def test_prior_count_matches_similar_task_count(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        mock_client.find_similar_tasks = AsyncMock(return_value=_similar_tasks(3))
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        await searcher.initialize_from_orcamind(_TASK_ID, simple_space)
        assert searcher._base.n_trials == 3

    async def test_zero_similar_tasks_injects_no_priors(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        mock_client.find_similar_tasks = AsyncMock(return_value=[])
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        await searcher.initialize_from_orcamind(_TASK_ID, simple_space)
        assert searcher._base.n_trials == 0

    async def test_fallback_to_predicted_score_when_no_accuracy_key(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        mock_client.find_similar_tasks = AsyncMock(return_value=_similar_tasks(1))
        mock_client.recommend_model = AsyncMock(return_value=_recommendation(predicted_score=0.77))
        mock_client.predict_performance = AsyncMock(return_value=_perf_metrics_no_accuracy())
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        await searcher.initialize_from_orcamind(_TASK_ID, simple_space)
        assert searcher._base.n_trials == 1
        best = searcher._base.get_best(1)
        assert best[0][1] == pytest.approx(0.77)

    async def test_prior_weight_scales_injected_scores(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        mock_client.find_similar_tasks = AsyncMock(return_value=_similar_tasks(1))
        mock_client.predict_performance = AsyncMock(return_value=_perf_metrics(accuracy=0.80))
        searcher = MetaInformedSearch(orcamind_client=mock_client, prior_weight=0.5)
        await searcher.initialize_from_orcamind(_TASK_ID, simple_space)
        best = searcher._base.get_best(1)
        assert best[0][1] == pytest.approx(0.80 * 0.5)

    async def test_top_k_priors_caps_injected_count(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        mock_client.find_similar_tasks = AsyncMock(return_value=_similar_tasks(10))
        searcher = MetaInformedSearch(orcamind_client=mock_client, top_k_priors=4)
        await searcher.initialize_from_orcamind(_TASK_ID, simple_space)
        assert searcher._base.n_trials == 4
