"""Tests for OrcaMindClient, OrcaLabClient, and OrcaNetClient.

All three are clients backed by httpx.AsyncClient. Tests cover:
  - Constructor URL normalisation
  - httpx Timeout and Limits configuration
  - HTTP behaviour for OrcaMindClient and OrcaLabClient (via respx mocking)
  - aclose() delegates to the underlying httpx client
  - Async context manager protocol
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from orca_shared.clients.orcalab_client import OrcaLabClient
from orca_shared.clients.orcamind_client import OrcaMindClient
from orca_shared.clients.orcanet_client import OrcaNetClient
from orca_shared.schemas.embedding import Embedding
from orca_shared.schemas.metrics import PerformanceMetrics
from orca_shared.schemas.model import ModelSummary
from orca_shared.schemas.recommendation import FeedbackRequest, ModelRecommendation, RecommendationRequest
from orca_shared.schemas.training import ExperimentResult, TrainingConfig

# Shared test UUIDs
TASK_ID = uuid.uuid4()
MODEL_ID = uuid.uuid4()
EXPERIMENT_ID = uuid.uuid4()
MAPPING_ID = uuid.uuid4()
NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# OrcaMindClient
# ---------------------------------------------------------------------------


class TestOrcaMindClientConstruction:
    def test_strips_trailing_slash(self):
        c = OrcaMindClient("http://host:8080/")
        assert c._base_url == "http://host:8080"

    def test_preserves_url_without_slash(self):
        c = OrcaMindClient("http://host:8080")
        assert c._base_url == "http://host:8080"

    def test_has_httpx_async_client(self):
        c = OrcaMindClient("http://host")
        assert isinstance(c._client, httpx.AsyncClient)

    def test_timeout_is_30_seconds(self):
        c = OrcaMindClient("http://host")
        assert c._client.timeout.read == pytest.approx(30.0)
        assert c._client.timeout.connect == pytest.approx(30.0)

    def test_max_connections_is_20(self):
        with patch("httpx.AsyncClient") as mock_cls:
            OrcaMindClient("http://host")
        limits = mock_cls.call_args[1]["limits"]
        assert limits.max_connections == 20


def _embedding_json(task_id: uuid.UUID = TASK_ID, dim: int = 4) -> dict:
    return {
        "embedding_id": str(uuid.uuid4()),
        "task_id": str(task_id),
        "embedding_type": "statistical",
        "embedding_vector": [0.1, 0.2, 0.3, 0.4],
        "dimension": dim,
        "model_version": "v1",
        "created_at": NOW.isoformat(),
    }


def _recommendation_json(task_id: uuid.UUID = TASK_ID, model_id: uuid.UUID = MODEL_ID) -> dict:
    return {
        "task_id": str(task_id),
        "model_id": str(model_id),
        "architecture": "resnet",
        "predicted_score": 0.85,
        "confidence": 0.9,
        "reasoning": None,
    }


class TestOrcaMindClientHTTP:
    @pytest.mark.asyncio
    async def test_recommend_model_returns_first_item(self):
        with respx.mock(base_url="http://host") as rm:
            rm.post("/api/v1/recommend-model").respond(200, json=[_recommendation_json()])
            c = OrcaMindClient("http://host")
            req = RecommendationRequest(task_embedding=[0.1, 0.2, 0.3, 0.4])
            result = await c.recommend_model(req)
        assert isinstance(result, ModelRecommendation)
        assert result.predicted_score == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_predict_performance_returns_metrics(self):
        with respx.mock(base_url="http://host") as rm:
            rm.post("/api/v1/predict-performance").respond(
                200,
                json={"model_id": str(MODEL_ID), "predicted_score": 0.77, "confidence": 0.88},
            )
            c = OrcaMindClient("http://host")
            result = await c.predict_performance([0.1, 0.2], MODEL_ID)
        assert isinstance(result, PerformanceMetrics)
        assert result.final_metrics["predicted_score"] == pytest.approx(0.77)
        assert result.final_metrics["confidence"] == pytest.approx(0.88)

    @pytest.mark.asyncio
    async def test_submit_feedback_succeeds_on_200(self):
        with respx.mock(base_url="http://host") as rm:
            rm.post("/api/v1/feedback").respond(200, json={"accepted": True})
            c = OrcaMindClient("http://host")
            req = FeedbackRequest(experiment_id=EXPERIMENT_ID, actual_metric=0.9, metric_name="f1")
            await c.submit_feedback(req)  # must not raise

    @pytest.mark.asyncio
    async def test_embed_task_returns_embedding(self):
        with respx.mock(base_url="http://host") as rm:
            rm.get(f"/api/v1/tasks/{TASK_ID}/embedding").respond(200, json=_embedding_json())
            c = OrcaMindClient("http://host")
            result = await c.embed_task(TASK_ID)
        assert isinstance(result, Embedding)
        assert len(result.embedding_vector) == 4

    @pytest.mark.asyncio
    async def test_find_similar_tasks_returns_list(self):
        payload = [
            {"task_id": str(uuid.uuid4()), "score": 0.9, "rank": 1},
            {"task_id": str(uuid.uuid4()), "score": 0.8, "rank": 2},
        ]
        with respx.mock(base_url="http://host") as rm:
            rm.post("/api/v1/similar-tasks").respond(200, json=payload)
            c = OrcaMindClient("http://host")
            results = await c.find_similar_tasks([0.1, 0.2, 0.3, 0.4], top_k=2)
        assert len(results) == 2
        assert results[0].score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_get_best_model_returns_model_summary(self):
        with respx.mock(base_url="http://host") as rm:
            rm.get(f"/api/v1/tasks/{TASK_ID}/embedding").respond(200, json=_embedding_json())
            rm.post("/api/v1/recommend-model").respond(200, json=[_recommendation_json()])
            c = OrcaMindClient("http://host")
            result = await c.get_best_model(TASK_ID)
        assert isinstance(result, ModelSummary)
        assert result.architecture == "resnet"

    @pytest.mark.asyncio
    async def test_recommend_model_raises_on_empty_list(self):
        with respx.mock(base_url="http://host") as rm:
            rm.post("/api/v1/recommend-model").respond(200, json=[])
            c = OrcaMindClient("http://host")
            with pytest.raises(ValueError, match="no model recommendations"):
                await c.recommend_model(RecommendationRequest(task_embedding=[0.1, 0.2]))

    @pytest.mark.asyncio
    async def test_recommend_model_raises_on_5xx(self):
        with respx.mock(base_url="http://host") as rm:
            rm.post("/api/v1/recommend-model").respond(503)
            c = OrcaMindClient("http://host")
            with pytest.raises(httpx.HTTPStatusError):
                await c.recommend_model(RecommendationRequest(task_embedding=[0.1]))

    @pytest.mark.asyncio
    async def test_submit_feedback_raises_on_4xx(self):
        with respx.mock(base_url="http://host") as rm:
            rm.post("/api/v1/feedback").respond(404)
            c = OrcaMindClient("http://host")
            req = FeedbackRequest(experiment_id=EXPERIMENT_ID, actual_metric=0.5, metric_name="f1")
            with pytest.raises(httpx.HTTPStatusError):
                await c.submit_feedback(req)

    @pytest.mark.asyncio
    async def test_embed_task_raises_on_404(self):
        with respx.mock(base_url="http://host") as rm:
            rm.get(f"/api/v1/tasks/{TASK_ID}/embedding").respond(404)
            c = OrcaMindClient("http://host")
            with pytest.raises(httpx.HTTPStatusError):
                await c.embed_task(TASK_ID)


class TestOrcaMindClientLifecycle:
    @pytest.mark.asyncio
    async def test_aclose_closes_httpx_client(self):
        c = OrcaMindClient("http://host")
        c._client = AsyncMock()
        await c.aclose()
        c._client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self):
        c = OrcaMindClient("http://host")
        c._client = AsyncMock()
        async with c as entered:
            assert entered is c

    @pytest.mark.asyncio
    async def test_context_manager_calls_aclose_on_exit(self):
        c = OrcaMindClient("http://host")
        c._client = AsyncMock()
        async with c:
            pass
        c._client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_manager_calls_aclose_on_exception(self):
        c = OrcaMindClient("http://host")
        c._client = AsyncMock()
        with pytest.raises(RuntimeError):
            async with c:
                raise RuntimeError("boom")
        c._client.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# OrcaLabClient
# ---------------------------------------------------------------------------


class TestOrcaLabClientConstruction:
    def test_strips_trailing_slash(self):
        c = OrcaLabClient("http://lab:9090/")
        assert c._base_url == "http://lab:9090"

    def test_has_httpx_async_client(self):
        assert isinstance(OrcaLabClient("http://lab")._client, httpx.AsyncClient)

    def test_timeout_is_30_seconds(self):
        c = OrcaLabClient("http://lab")
        assert c._client.timeout.read == pytest.approx(30.0)

    def test_max_connections_is_20(self):
        with patch("httpx.AsyncClient") as mock_cls:
            OrcaLabClient("http://lab")
        limits = mock_cls.call_args[1]["limits"]
        assert limits.max_connections == 20


def _experiment_json(experiment_id: uuid.UUID = EXPERIMENT_ID, status: str = "COMPLETED") -> dict:
    return {
        "experiment_id": str(experiment_id),
        "task_id": str(TASK_ID),
        "model_id": str(MODEL_ID),
        "status": status,
        "mlflow_run_id": None,
        "started_at": NOW.isoformat(),
        "completed_at": NOW.isoformat(),
        "metrics": {"accuracy": 0.91},
    }


class TestOrcaLabClientHTTP:
    @pytest.mark.asyncio
    async def test_create_experiment_posts_and_returns_id(self):
        with respx.mock(base_url="http://lab") as rm:
            rm.post("/api/v1/experiments").respond(
                200, json={"experiment_id": str(EXPERIMENT_ID)}
            )
            c = OrcaLabClient("http://lab")
            result = await c.create_experiment(
                task_id=str(TASK_ID),
                model_config={"model_id": str(MODEL_ID), "name": "resnet18"},
                tags=["transfer_validation"],
            )
        assert result == str(EXPERIMENT_ID)

    @pytest.mark.asyncio
    async def test_create_experiment_includes_model_id_in_payload(self):
        with respx.mock(base_url="http://lab") as rm:
            route = rm.post("/api/v1/experiments").respond(
                200, json={"experiment_id": str(EXPERIMENT_ID)}
            )
            c = OrcaLabClient("http://lab")
            await c.create_experiment(
                task_id=str(TASK_ID),
                model_config={"model_id": str(MODEL_ID), "name": "resnet18"},
                tags=[],
            )
        sent = route.calls[0].request
        import json as _json
        body = _json.loads(sent.content)
        assert body["model_id"] == str(MODEL_ID)
        assert body["task_id"] == str(TASK_ID)

    @pytest.mark.asyncio
    async def test_create_experiment_accepts_pydantic_model(self):
        from orca_shared.schemas.model import ModelSummary as MS
        model = MS(model_id=MODEL_ID, name="vgg", architecture="vgg")
        with respx.mock(base_url="http://lab") as rm:
            rm.post("/api/v1/experiments").respond(
                200, json={"experiment_id": str(EXPERIMENT_ID)}
            )
            c = OrcaLabClient("http://lab")
            result = await c.create_experiment(
                task_id=str(TASK_ID),
                model_config=model,
                tags=["test"],
            )
        assert result == str(EXPERIMENT_ID)

    @pytest.mark.asyncio
    async def test_create_experiment_raises_on_4xx(self):
        with respx.mock(base_url="http://lab") as rm:
            rm.post("/api/v1/experiments").respond(422)
            c = OrcaLabClient("http://lab")
            with pytest.raises(httpx.HTTPStatusError):
                await c.create_experiment(str(TASK_ID), {}, [])

    @pytest.mark.asyncio
    async def test_wait_for_completion_returns_on_completed_status(self):
        with respx.mock(base_url="http://lab") as rm:
            rm.get(f"/api/v1/experiments/{EXPERIMENT_ID}").respond(
                200, json=_experiment_json(status="COMPLETED")
            )
            c = OrcaLabClient("http://lab")
            result = await c.wait_for_completion(str(EXPERIMENT_ID), timeout=60, poll_interval=1)
        assert isinstance(result, ExperimentResult)
        assert result.status == "COMPLETED"

    @pytest.mark.asyncio
    async def test_wait_for_completion_returns_on_failed_status(self):
        with respx.mock(base_url="http://lab") as rm:
            rm.get(f"/api/v1/experiments/{EXPERIMENT_ID}").respond(
                200, json=_experiment_json(status="FAILED")
            )
            c = OrcaLabClient("http://lab")
            result = await c.wait_for_completion(str(EXPERIMENT_ID), timeout=60, poll_interval=1)
        assert result.status == "FAILED"

    @pytest.mark.asyncio
    async def test_wait_for_completion_polls_until_terminal(self):
        """Verify that polling continues through non-terminal statuses."""
        responses = [
            _experiment_json(status="running"),
            _experiment_json(status="running"),
            _experiment_json(status="COMPLETED"),
        ]
        call_count = 0

        async def _side_effect(request, *args, **kwargs):
            nonlocal call_count
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return httpx.Response(200, json=resp)

        with respx.mock(base_url="http://lab") as rm:
            rm.get(f"/api/v1/experiments/{EXPERIMENT_ID}").mock(side_effect=_side_effect)
            c = OrcaLabClient("http://lab")
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await c.wait_for_completion(
                    str(EXPERIMENT_ID), timeout=300, poll_interval=1
                )
        assert result.status == "COMPLETED"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_wait_for_completion_raises_timeout_error(self):
        """TimeoutError raised when deadline passes with non-terminal status."""
        with respx.mock(base_url="http://lab") as rm:
            rm.get(f"/api/v1/experiments/{EXPERIMENT_ID}").respond(
                200, json=_experiment_json(status="running")
            )
            c = OrcaLabClient("http://lab")
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(TimeoutError, match=str(EXPERIMENT_ID)):
                    await c.wait_for_completion(
                        str(EXPERIMENT_ID), timeout=0, poll_interval=1
                    )


class TestOrcaLabClientStubs:
    @pytest.mark.asyncio
    async def test_start_sweep_raises(self):
        c = OrcaLabClient("http://lab")
        with pytest.raises(NotImplementedError):
            await c.start_sweep(EXPERIMENT_ID, {"lr": [1e-3, 1e-4]})

    @pytest.mark.asyncio
    async def test_get_sweep_status_raises(self):
        c = OrcaLabClient("http://lab")
        with pytest.raises(NotImplementedError):
            await c.get_sweep_status("sweep-xyz")


class TestOrcaLabClientLifecycle:
    @pytest.mark.asyncio
    async def test_aclose_closes_httpx_client(self):
        c = OrcaLabClient("http://lab")
        c._client = AsyncMock()
        await c.aclose()
        c._client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self):
        c = OrcaLabClient("http://lab")
        c._client = AsyncMock()
        async with c as entered:
            assert entered is c

    @pytest.mark.asyncio
    async def test_context_manager_calls_aclose_on_exit(self):
        c = OrcaLabClient("http://lab")
        c._client = AsyncMock()
        async with c:
            pass
        c._client.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# OrcaNetClient
# ---------------------------------------------------------------------------


class TestOrcaNetClientConstruction:
    def test_strips_trailing_slash(self):
        c = OrcaNetClient("http://net:7070/")
        assert c._base_url == "http://net:7070"

    def test_has_httpx_async_client(self):
        assert isinstance(OrcaNetClient("http://net")._client, httpx.AsyncClient)

    def test_timeout_is_30_seconds(self):
        c = OrcaNetClient("http://net")
        assert c._client.timeout.read == pytest.approx(30.0)

    def test_max_connections_is_20(self):
        with patch("httpx.AsyncClient") as mock_cls:
            OrcaNetClient("http://net")
        limits = mock_cls.call_args[1]["limits"]
        assert limits.max_connections == 20


class TestOrcaNetClientStubs:
    @pytest.mark.asyncio
    async def test_score_transfer_raises(self):
        c = OrcaNetClient("http://net")
        with pytest.raises(NotImplementedError):
            await c.score_transfer(TASK_ID, uuid.uuid4())

    @pytest.mark.asyncio
    async def test_recommend_transfer_raises(self):
        c = OrcaNetClient("http://net")
        with pytest.raises(NotImplementedError):
            await c.recommend_transfer(TASK_ID)

    @pytest.mark.asyncio
    async def test_explain_transfer_raises(self):
        c = OrcaNetClient("http://net")
        with pytest.raises(NotImplementedError):
            await c.explain_transfer(MAPPING_ID)


class TestOrcaNetClientLifecycle:
    @pytest.mark.asyncio
    async def test_aclose_closes_httpx_client(self):
        c = OrcaNetClient("http://net")
        c._client = AsyncMock()
        await c.aclose()
        c._client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self):
        c = OrcaNetClient("http://net")
        c._client = AsyncMock()
        async with c as entered:
            assert entered is c

    @pytest.mark.asyncio
    async def test_context_manager_calls_aclose_on_exit(self):
        c = OrcaNetClient("http://net")
        c._client = AsyncMock()
        async with c:
            pass
        c._client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_manager_calls_aclose_on_exception(self):
        c = OrcaNetClient("http://net")
        c._client = AsyncMock()
        with pytest.raises(ValueError):
            async with c:
                raise ValueError("transfer failed")
        c._client.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# Cross-client: verify all have expected method names
# ---------------------------------------------------------------------------


class TestClientMethodSurface:
    def test_orcamind_has_all_methods(self):
        c = OrcaMindClient("http://h")
        for method in (
            "recommend_model", "predict_performance", "submit_feedback",
            "get_best_model", "embed_task", "find_similar_tasks", "aclose",
        ):
            assert callable(getattr(c, method)), f"Missing method: {method}"

    def test_orcalab_has_all_methods(self):
        c = OrcaLabClient("http://h")
        for method in (
            "create_experiment", "start_sweep", "get_sweep_status",
            "wait_for_completion", "aclose",
        ):
            assert callable(getattr(c, method)), f"Missing method: {method}"

    def test_orcanet_has_all_methods(self):
        c = OrcaNetClient("http://h")
        for method in ("score_transfer", "recommend_transfer", "explain_transfer", "aclose"):
            assert callable(getattr(c, method)), f"Missing method: {method}"
