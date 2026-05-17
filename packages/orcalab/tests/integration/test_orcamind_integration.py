"""Integration tests for OrcaLab ↔ OrcaMind bidirectional flows.

OrcaMind's HTTP API is mocked with respx so no running service is needed.
Tests validate:
  - MetaInformedSearch warm-start with mocked priors
  - Graceful fallback on OrcaMind 5xx and network errors
  - submit_feedback called for every completed trial after a sweep
  - Feedback payloads carry correct task/metric/params fields
  - get_orcamind_priors Prefect task happy path and network-error path
  - log_results task submits feedback and swallows HTTP errors
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from orca_shared.clients.orcamind_client import OrcaMindClient
from orca_shared.schemas.recommendation import FeedbackRequest, ModelRecommendation
from orca_shared.schemas.training import ExperimentResult
from orcalab.orchestration.tasks.get_priors import get_orcamind_priors
from orcalab.orchestration.tasks.log_results import log_results
from orcalab.search.meta_informed import MetaInformedSearch
from orcalab.search_spaces.parameters import FloatParameter, IntParameter
from orcalab.search_spaces.space import SearchSpace

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ORCAMIND_URL = "http://localhost:8001"
_TASK_UUID = uuid.uuid4()
_TASK_ID = str(_TASK_UUID)
_MODEL_UUID = uuid.uuid4()
_EMBED_ID = uuid.uuid4()
_NOW = datetime.now(timezone.utc)

# ---------------------------------------------------------------------------
# Shared JSON helpers
# ---------------------------------------------------------------------------


def _embedding_json(dim: int = 4) -> dict:
    return {
        "embedding_id": str(_EMBED_ID),
        "task_id": str(_TASK_UUID),
        "embedding_type": "statistical",
        "embedding_vector": [0.1] * dim,
        "dimension": dim,
        "model_version": "v1",
        "created_at": _NOW.isoformat(),
    }


def _recommendation_json(predicted_score: float = 0.85) -> dict:
    return {
        "task_id": str(_TASK_UUID),
        "model_id": str(_MODEL_UUID),
        "architecture": "resnet",
        "predicted_score": predicted_score,
        "confidence": 0.9,
        "reasoning": None,
    }


def _similar_tasks_json(n: int = 3) -> list[dict]:
    return [
        {"task_id": str(uuid.uuid4()), "score": 0.9 - i * 0.1, "rank": i + 1}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def search_space() -> SearchSpace:
    return (
        SearchSpace("integ_test")
        .add(FloatParameter("lr", low=1e-4, high=1e-1))
        .add(IntParameter("layers", low=2, high=6))
    )


@pytest.fixture
def mock_orcamind():
    """respx router with all standard OrcaMind endpoints pre-wired.

    Named routes are accessible via mock_orcamind["embed_task"] etc.
    assert_all_called=False because individual tests may only exercise a subset.
    """
    with respx.mock(base_url=_ORCAMIND_URL, assert_all_called=False) as rm:
        rm.get(f"/api/v1/tasks/{_TASK_UUID}/embedding", name="embed_task").respond(
            200, json=_embedding_json()
        )
        rm.post("/api/v1/recommend-model", name="recommend_model").respond(
            200, json=[_recommendation_json()]
        )
        rm.post("/api/v1/similar-tasks", name="similar_tasks").respond(
            200, json=_similar_tasks_json(3)
        )
        rm.post("/api/v1/feedback", name="feedback").respond(
            200, json={"accepted": True}
        )
        yield rm


# ---------------------------------------------------------------------------
# MetaInformedSearch integration
# ---------------------------------------------------------------------------


class TestMetaInformedSweepWithOrcaMind:
    async def test_priors_injected_into_base_strategy(
        self, mock_orcamind, search_space: SearchSpace
    ) -> None:
        async with OrcaMindClient(_ORCAMIND_URL) as client:
            strategy = MetaInformedSearch(orcamind_client=client)
            await strategy.initialize_from_orcamind(_TASK_ID, search_space)
        assert strategy._base.n_trials == 3

    async def test_all_three_orcamind_endpoints_called(
        self, mock_orcamind, search_space: SearchSpace
    ) -> None:
        async with OrcaMindClient(_ORCAMIND_URL) as client:
            strategy = MetaInformedSearch(orcamind_client=client)
            await strategy.initialize_from_orcamind(_TASK_ID, search_space)
        assert mock_orcamind["embed_task"].called
        assert mock_orcamind["recommend_model"].called
        assert mock_orcamind["similar_tasks"].called

    async def test_suggest_works_after_warm_start(
        self, mock_orcamind, search_space: SearchSpace
    ) -> None:
        async with OrcaMindClient(_ORCAMIND_URL) as client:
            strategy = MetaInformedSearch(orcamind_client=client)
            await strategy.initialize_from_orcamind(_TASK_ID, search_space)
            params = strategy.suggest(search_space)
        assert "lr" in params and "layers" in params


class TestMetaInformedGracefulDegradation:
    async def test_fallback_on_5xx_recommend(self, search_space: SearchSpace) -> None:
        with respx.mock(base_url=_ORCAMIND_URL, assert_all_called=False) as rm:
            rm.get(f"/api/v1/tasks/{_TASK_UUID}/embedding").respond(
                200, json=_embedding_json()
            )
            rm.post("/api/v1/recommend-model").respond(503)

            async with OrcaMindClient(_ORCAMIND_URL) as client:
                strategy = MetaInformedSearch(orcamind_client=client)
                await strategy.initialize_from_orcamind(_TASK_ID, search_space)

        assert strategy._base.n_trials == 0

    async def test_sweep_continues_after_5xx(self, search_space: SearchSpace) -> None:
        with respx.mock(base_url=_ORCAMIND_URL, assert_all_called=False) as rm:
            rm.get(f"/api/v1/tasks/{_TASK_UUID}/embedding").respond(200, json=_embedding_json())
            rm.post("/api/v1/recommend-model").respond(503)

            async with OrcaMindClient(_ORCAMIND_URL) as client:
                strategy = MetaInformedSearch(orcamind_client=client)
                await strategy.initialize_from_orcamind(_TASK_ID, search_space)
                params = strategy.suggest(search_space)
                strategy.update(params, result=0.7)

        assert strategy.n_trials == 1

    async def test_fallback_on_connect_error(self, search_space: SearchSpace) -> None:
        with respx.mock(base_url=_ORCAMIND_URL, assert_all_called=False) as rm:
            rm.get(f"/api/v1/tasks/{_TASK_UUID}/embedding").mock(
                side_effect=httpx.ConnectError("refused")
            )
            async with OrcaMindClient(_ORCAMIND_URL) as client:
                strategy = MetaInformedSearch(orcamind_client=client)
                await strategy.initialize_from_orcamind(_TASK_ID, search_space)

        assert strategy._base.n_trials == 0


# ---------------------------------------------------------------------------
# Feedback / flush results
# ---------------------------------------------------------------------------


class TestFlushResultsToOrcaMind:
    async def test_submit_feedback_called_per_trial(
        self, mock_orcamind, search_space: SearchSpace
    ) -> None:
        async with OrcaMindClient(_ORCAMIND_URL) as client:
            strategy = MetaInformedSearch(orcamind_client=client)
            for _ in range(4):
                p = strategy.suggest(search_space)
                strategy.update(p, result=0.8)
            await strategy.flush_results_to_orcamind(_TASK_ID)

        assert mock_orcamind["feedback"].call_count == 4

    async def test_feedback_payload_has_correct_fields(
        self, mock_orcamind, search_space: SearchSpace
    ) -> None:
        target_metric = 0.77
        async with OrcaMindClient(_ORCAMIND_URL) as client:
            strategy = MetaInformedSearch(orcamind_client=client)
            p = strategy.suggest(search_space)
            strategy.update(p, result=target_metric)
            await strategy.flush_results_to_orcamind(_TASK_ID)

        last_request = mock_orcamind["feedback"].calls.last.request
        body = json.loads(last_request.content)
        assert body["metric_name"] == "objective"
        assert body["actual_metric"] == pytest.approx(target_metric)
        assert "experiment_id" in body
        assert "lr" in (body.get("params") or {})

    async def test_no_feedback_sent_when_no_trials(
        self, mock_orcamind
    ) -> None:
        async with OrcaMindClient(_ORCAMIND_URL) as client:
            strategy = MetaInformedSearch(orcamind_client=client)
            await strategy.flush_results_to_orcamind(_TASK_ID)

        assert mock_orcamind["feedback"].call_count == 0


# ---------------------------------------------------------------------------
# get_orcamind_priors task
# ---------------------------------------------------------------------------


class TestGetOrcaMindPriorsTask:
    async def test_returns_recommendation_list(self, mock_orcamind) -> None:
        result = await get_orcamind_priors.fn(_TASK_ID, _ORCAMIND_URL)
        assert result is not None
        assert len(result) == 1
        assert isinstance(result[0], ModelRecommendation)
        assert result[0].predicted_score == pytest.approx(0.85)

    async def test_embed_and_recommend_endpoints_both_called(self, mock_orcamind) -> None:
        await get_orcamind_priors.fn(_TASK_ID, _ORCAMIND_URL)
        assert mock_orcamind["embed_task"].called
        assert mock_orcamind["recommend_model"].called

    async def test_returns_none_on_connect_error(self) -> None:
        with respx.mock(base_url=_ORCAMIND_URL, assert_all_called=False) as rm:
            rm.get(f"/api/v1/tasks/{_TASK_UUID}/embedding").mock(
                side_effect=httpx.ConnectError("refused")
            )
            result = await get_orcamind_priors.fn(_TASK_ID, _ORCAMIND_URL)
        assert result is None

    async def test_returns_none_on_timeout(self) -> None:
        with respx.mock(base_url=_ORCAMIND_URL, assert_all_called=False) as rm:
            rm.get(f"/api/v1/tasks/{_TASK_UUID}/embedding").mock(
                side_effect=httpx.TimeoutException("timed out")
            )
            result = await get_orcamind_priors.fn(_TASK_ID, _ORCAMIND_URL)
        assert result is None

    async def test_has_retry_decorator(self) -> None:
        assert get_orcamind_priors.retries == 1  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# log_results task
# ---------------------------------------------------------------------------


class TestLogResultsTask:
    def _make_result(self, metrics: dict | None = None) -> ExperimentResult:
        return ExperimentResult(
            experiment_id=uuid.uuid4(),
            status="completed",
            metrics=metrics or {"accuracy": 0.9, "loss": 0.1},
        )

    async def test_calls_submit_feedback_once(self, mock_orcamind) -> None:
        result = self._make_result()
        async with OrcaMindClient(_ORCAMIND_URL) as client:
            await log_results.fn(result, client)
        assert mock_orcamind["feedback"].call_count == 1

    async def test_feedback_uses_max_metric(self, mock_orcamind) -> None:
        result = self._make_result({"accuracy": 0.9, "f1": 0.75})
        async with OrcaMindClient(_ORCAMIND_URL) as client:
            await log_results.fn(result, client)

        body = json.loads(mock_orcamind["feedback"].calls.last.request.content)
        assert body["actual_metric"] == pytest.approx(0.9)

    async def test_swallows_http_status_error(self) -> None:
        result = self._make_result()
        client = AsyncMock()
        client.submit_feedback = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "503", request=AsyncMock(), response=AsyncMock()
            )
        )
        await log_results.fn(result, client)  # must not raise

    async def test_swallows_connect_error(self) -> None:
        result = self._make_result()
        client = AsyncMock()
        client.submit_feedback = AsyncMock(
            side_effect=httpx.ConnectError("refused")
        )
        await log_results.fn(result, client)  # must not raise

    async def test_swallows_timeout_error(self) -> None:
        result = self._make_result()
        client = AsyncMock()
        client.submit_feedback = AsyncMock(
            side_effect=httpx.TimeoutException("timed out")
        )
        await log_results.fn(result, client)  # must not raise
