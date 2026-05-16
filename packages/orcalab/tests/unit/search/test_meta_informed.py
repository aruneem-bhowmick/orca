"""Unit tests for MetaInformedSearch."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from orca_shared.clients.orcamind_client import OrcaMindClient
from orca_shared.schemas.embedding import Embedding, SimilarityResult
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock(spec=OrcaMindClient)
    client.embed_task = AsyncMock(return_value=_embedding())
    client.recommend_model = AsyncMock(return_value=_recommendation())
    client.find_similar_tasks = AsyncMock(return_value=_similar_tasks(3))
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

    async def test_scores_are_product_of_similarity_and_predicted_score(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        # similarity=0.9, predicted_score=0.77 → expected prior score = 0.9 * 0.77
        mock_client.find_similar_tasks = AsyncMock(
            return_value=[SimilarityResult(task_id=uuid.uuid4(), score=0.9, rank=1)]
        )
        mock_client.recommend_model = AsyncMock(return_value=_recommendation(predicted_score=0.77))
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        await searcher.initialize_from_orcamind(_TASK_ID, simple_space)
        assert searcher._base.n_trials == 1
        best = searcher._base.get_best(1)
        assert best[0][1] == pytest.approx(0.9 * 0.77)

    async def test_prior_weight_scales_injected_scores(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        # similarity=1.0, predicted_score=0.80, prior_weight=0.5 → score = 0.80 * 0.5
        mock_client.find_similar_tasks = AsyncMock(
            return_value=[SimilarityResult(task_id=uuid.uuid4(), score=1.0, rank=1)]
        )
        mock_client.recommend_model = AsyncMock(return_value=_recommendation(predicted_score=0.80))
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


# ---------------------------------------------------------------------------
# TestGracefulDegradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    async def test_connect_error_does_not_raise(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        mock_client.embed_task = AsyncMock(side_effect=httpx.ConnectError("refused"))
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        await searcher.initialize_from_orcamind(_TASK_ID, simple_space)
        assert searcher._base.n_trials == 0

    async def test_connect_error_sweep_still_works(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        mock_client.embed_task = AsyncMock(side_effect=httpx.ConnectError("refused"))
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        await searcher.initialize_from_orcamind(_TASK_ID, simple_space)
        params = searcher.suggest(simple_space)
        searcher.update(params, result=0.7)
        assert searcher.n_trials == 1

    async def test_timeout_error_does_not_raise(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        mock_client.find_similar_tasks = AsyncMock(
            side_effect=httpx.TimeoutException("timed out")
        )
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        await searcher.initialize_from_orcamind(_TASK_ID, simple_space)
        assert searcher._base.n_trials == 0

    async def test_http_status_error_does_not_raise(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        response = MagicMock()
        response.status_code = 503
        mock_client.recommend_model = AsyncMock(
            side_effect=httpx.HTTPStatusError("503", request=MagicMock(), response=response)
        )
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        await searcher.initialize_from_orcamind(_TASK_ID, simple_space)
        assert searcher._base.n_trials == 0

    async def test_inject_priors_never_called_on_network_error(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        mock_client.embed_task = AsyncMock(side_effect=httpx.ConnectError("refused"))
        base = MagicMock(spec=BayesianSearch)
        base.inject_priors = MagicMock()
        searcher = MetaInformedSearch(orcamind_client=mock_client, base_strategy=base)
        await searcher.initialize_from_orcamind(_TASK_ID, simple_space)
        base.inject_priors.assert_not_called()


# ---------------------------------------------------------------------------
# TestSuggestUpdate
# ---------------------------------------------------------------------------


class TestSuggestUpdate:
    def test_suggest_returns_valid_param_dict(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        params = searcher.suggest(simple_space)
        assert "lr" in params
        assert "layers" in params

    def test_update_increments_n_trials(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        params = searcher.suggest(simple_space)
        searcher.update(params, result=0.8)
        assert searcher.n_trials == 1

    def test_nan_result_does_not_count_as_trial(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        params = searcher.suggest(simple_space)
        searcher.update(params, result=float("nan"))
        assert searcher.n_trials == 0

    def test_nan_result_not_stored_in_completed_results(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        params = searcher.suggest(simple_space)
        searcher.update(params, result=float("nan"))
        assert searcher._completed_results == []

    def test_inf_result_does_not_count_as_trial(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        params = searcher.suggest(simple_space)
        searcher.update(params, result=float("inf"))
        assert searcher.n_trials == 0

    def test_neg_inf_result_not_stored_in_completed_results(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        params = searcher.suggest(simple_space)
        searcher.update(params, result=float("-inf"))
        assert searcher._completed_results == []

    def test_get_best_returns_descending_order(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        for score in [0.3, 0.9, 0.6]:
            p = searcher.suggest(simple_space)
            searcher.update(p, result=score)
        best = searcher.get_best(3)
        values = [v for _, v in best]
        assert values == sorted(values, reverse=True)
        assert values[0] == pytest.approx(0.9)

    def test_n_trials_delegates_to_base(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        base = BayesianSearch()
        searcher = MetaInformedSearch(orcamind_client=mock_client, base_strategy=base)
        for _ in range(5):
            p = searcher.suggest(simple_space)
            searcher.update(p, result=0.5)
        assert searcher.n_trials == base.n_trials == 5


# ---------------------------------------------------------------------------
# TestFlushResults
# ---------------------------------------------------------------------------


class TestFlushResults:
    async def test_submit_feedback_called_once_per_finite_result(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        for _ in range(3):
            p = searcher.suggest(simple_space)
            searcher.update(p, result=0.8)
        await searcher.flush_results_to_orcamind(_TASK_ID)
        assert mock_client.submit_feedback.call_count == 3

    async def test_nan_result_excluded_from_flush(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        bad = searcher.suggest(simple_space)
        searcher.update(bad, result=float("nan"))
        for _ in range(2):
            p = searcher.suggest(simple_space)
            searcher.update(p, result=0.7)
        await searcher.flush_results_to_orcamind(_TASK_ID)
        assert mock_client.submit_feedback.call_count == 2

    async def test_inf_result_excluded_from_flush(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        bad = searcher.suggest(simple_space)
        searcher.update(bad, result=float("inf"))
        good = searcher.suggest(simple_space)
        searcher.update(good, result=0.7)
        await searcher.flush_results_to_orcamind(_TASK_ID)
        assert mock_client.submit_feedback.call_count == 1

    async def test_neg_inf_result_excluded_from_flush(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        bad = searcher.suggest(simple_space)
        searcher.update(bad, result=float("-inf"))
        good = searcher.suggest(simple_space)
        searcher.update(good, result=0.6)
        await searcher.flush_results_to_orcamind(_TASK_ID)
        assert mock_client.submit_feedback.call_count == 1

    async def test_flush_sends_correct_metric_values(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        expected_results = [0.6, 0.75, 0.9]
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        for score in expected_results:
            p = searcher.suggest(simple_space)
            searcher.update(p, result=score)
        await searcher.flush_results_to_orcamind(_TASK_ID)
        sent_metrics = [
            call_args.args[0].actual_metric
            for call_args in mock_client.submit_feedback.call_args_list
        ]
        assert sent_metrics == pytest.approx(expected_results)

    async def test_flush_with_no_completed_results_is_noop(
        self, mock_client: MagicMock
    ) -> None:
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        await searcher.flush_results_to_orcamind(_TASK_ID)
        mock_client.submit_feedback.assert_not_called()

    async def test_flush_sends_feedback_request_objects(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        p = searcher.suggest(simple_space)
        searcher.update(p, result=0.8)
        await searcher.flush_results_to_orcamind(_TASK_ID)
        call_arg = mock_client.submit_feedback.call_args.args[0]
        assert isinstance(call_arg, FeedbackRequest)
        assert call_arg.metric_name == "objective"
        assert call_arg.actual_metric == pytest.approx(0.8)
        assert call_arg.params is not None
        assert "lr" in call_arg.params
        assert "layers" in call_arg.params

    async def test_second_flush_does_not_resubmit_same_results(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        p = searcher.suggest(simple_space)
        searcher.update(p, result=0.8)

        await searcher.flush_results_to_orcamind(_TASK_ID)
        first_count = mock_client.submit_feedback.call_count

        await searcher.flush_results_to_orcamind(_TASK_ID)
        assert mock_client.submit_feedback.call_count == first_count

    async def test_flush_network_error_does_not_propagate(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        mock_client.submit_feedback = AsyncMock(
            side_effect=httpx.ConnectError("refused")
        )
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        p = searcher.suggest(simple_space)
        searcher.update(p, result=0.8)
        await searcher.flush_results_to_orcamind(_TASK_ID)  # must not raise

    async def test_flush_error_preserves_results_for_retry(
        self, mock_client: MagicMock, simple_space: SearchSpace
    ) -> None:
        mock_client.submit_feedback = AsyncMock(
            side_effect=httpx.ConnectError("refused")
        )
        searcher = MetaInformedSearch(orcamind_client=mock_client)
        p = searcher.suggest(simple_space)
        searcher.update(p, result=0.8)
        await searcher.flush_results_to_orcamind(_TASK_ID)
        assert len(searcher._completed_results) == 1
