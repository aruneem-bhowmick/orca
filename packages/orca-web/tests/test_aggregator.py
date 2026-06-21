"""Tests for orca_web.services.aggregator."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from orca_web.services.aggregator import Aggregator


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch):
    s = SimpleNamespace(
        orcamind_api_url="http://mind:8000",
        orcalab_api_url="http://lab:8001",
        orcanet_api_url="http://net:8002",
    )
    monkeypatch.setattr("orca_web.services.aggregator.settings", s)


def _mock_client(responses: dict[str, dict | list | Exception]):
    """Build an AsyncMock httpx client that returns canned responses by URL substring."""
    client = AsyncMock(spec=httpx.AsyncClient)

    async def fake_get(url, **kwargs):
        for pattern, val in responses.items():
            if pattern in url:
                if isinstance(val, Exception):
                    raise val
                resp = MagicMock()
                resp.json.return_value = val
                resp.raise_for_status = MagicMock()
                return resp
        resp = MagicMock()
        resp.json.return_value = {}
        resp.raise_for_status = MagicMock()
        return resp

    client.get = AsyncMock(side_effect=fake_get)
    return client


class TestSafeGet:
    async def test_returns_json_on_success(self):
        client = _mock_client({"/test": {"ok": True}})
        agg = Aggregator(client)
        result = await agg._safe_get("http://x/test")
        assert result == {"ok": True}

    async def test_returns_empty_dict_on_exception(self):
        client = _mock_client({"/fail": httpx.ConnectError("down")})
        agg = Aggregator(client)
        result = await agg._safe_get("http://x/fail")
        assert result == {}

    async def test_returns_empty_dict_on_http_error(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        resp = MagicMock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        client.get = AsyncMock(return_value=resp)
        agg = Aggregator(client)
        result = await agg._safe_get("http://x/err")
        assert result == {}


class TestOverview:
    async def test_aggregates_all_services(self):
        client = _mock_client({
            "tasks?limit=5": [{"id": 1}],
            "experiments?limit=5": [{"id": 2}],
            "mind:8000/health": {"status": "healthy"},
            "lab:8001/health": {"status": "healthy"},
            "net:8002/health": {"status": "ok"},
        })
        agg = Aggregator(client)
        result = await agg.overview()
        assert result["recent_tasks"] == [{"id": 1}]
        assert result["recent_experiments"] == [{"id": 2}]
        assert result["services"]["orcamind"] == "healthy"
        assert result["services"]["orcalab"] == "healthy"
        assert result["services"]["orcanet"] == "ok"

    async def test_graceful_on_service_failure(self):
        client = _mock_client({
            "tasks": httpx.ConnectError("down"),
            "experiments": httpx.ConnectError("down"),
            "health": {},
        })
        agg = Aggregator(client)
        result = await agg.overview()
        assert result["recent_tasks"] == []
        assert result["recent_experiments"] == []
        assert result["services"]["orcamind"] == "unknown"


class TestPublicStats:
    async def test_returns_counts(self):
        client = _mock_client({
            "tasks?limit=1": [{"id": 1}],
            "experiments?limit=1": [{"id": 2}],
        })
        agg = Aggregator(client)
        result = await agg.public_stats()
        assert result["task_count"] == 1
        assert result["experiment_count"] == 1

    async def test_returns_zero_on_failure(self):
        client = _mock_client({
            "tasks": httpx.ConnectError("down"),
            "experiments": httpx.ConnectError("down"),
        })
        agg = Aggregator(client)
        result = await agg.public_stats()
        assert result["task_count"] == 0
        assert result["experiment_count"] == 0
