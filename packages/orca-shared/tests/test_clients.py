"""Tests for OrcaMindClient, OrcaLabClient, and OrcaNetClient.

All three are stub clients backed by httpx.AsyncClient. Tests cover:
  - Constructor URL normalisation
  - httpx Timeout and Limits configuration
  - Every stub method raises NotImplementedError
  - aclose() delegates to the underlying httpx client
  - Async context manager protocol
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from orca_shared.clients.orcalab_client import OrcaLabClient
from orca_shared.clients.orcamind_client import OrcaMindClient
from orca_shared.clients.orcanet_client import OrcaNetClient
from orca_shared.schemas.embedding import Embedding
from orca_shared.schemas.metrics import PerformanceMetrics
from orca_shared.schemas.recommendation import FeedbackRequest, RecommendationRequest
from orca_shared.schemas.training import TrainingConfig

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


class TestOrcaMindClientStubs:
    @pytest.mark.asyncio
    async def test_recommend_model_raises(self):
        c = OrcaMindClient("http://host")
        req = RecommendationRequest(task_embedding=[0.1, 0.2])
        with pytest.raises(NotImplementedError):
            await c.recommend_model(req)

    @pytest.mark.asyncio
    async def test_predict_performance_raises(self):
        c = OrcaMindClient("http://host")
        with pytest.raises(NotImplementedError):
            await c.predict_performance([0.1], MODEL_ID)

    @pytest.mark.asyncio
    async def test_submit_feedback_raises(self):
        c = OrcaMindClient("http://host")
        req = FeedbackRequest(experiment_id=EXPERIMENT_ID, actual_metric=0.9, metric_name="f1")
        with pytest.raises(NotImplementedError):
            await c.submit_feedback(req)

    @pytest.mark.asyncio
    async def test_get_best_model_raises(self):
        c = OrcaMindClient("http://host")
        with pytest.raises(NotImplementedError):
            await c.get_best_model(TASK_ID)

    @pytest.mark.asyncio
    async def test_embed_task_raises(self):
        c = OrcaMindClient("http://host")
        with pytest.raises(NotImplementedError):
            await c.embed_task(TASK_ID)

    @pytest.mark.asyncio
    async def test_find_similar_tasks_raises(self):
        c = OrcaMindClient("http://host")
        with pytest.raises(NotImplementedError):
            await c.find_similar_tasks([0.1, 0.2])


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


class TestOrcaLabClientStubs:
    @pytest.mark.asyncio
    async def test_create_experiment_raises(self):
        c = OrcaLabClient("http://lab")
        cfg = TrainingConfig()
        with pytest.raises(NotImplementedError):
            await c.create_experiment(TASK_ID, MODEL_ID, cfg)

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

    @pytest.mark.asyncio
    async def test_wait_for_completion_raises(self):
        c = OrcaLabClient("http://lab")
        with pytest.raises(NotImplementedError):
            await c.wait_for_completion("sweep-xyz")


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
