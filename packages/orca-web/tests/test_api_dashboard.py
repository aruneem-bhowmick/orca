"""Tests for orca_web.api.routers.dashboard endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from orca_web.api.routers.dashboard import overview, public_stats
from orca_web.services.aggregator import Aggregator


class TestOverview:
    async def test_returns_aggregated_data(self, user_factory):
        user = user_factory()
        agg = AsyncMock(spec=Aggregator)
        agg.overview = AsyncMock(
            return_value={"orcamind": {"tasks": 5}, "orcalab": {"experiments": 3}, "orcanet": {}}
        )
        result = await overview(_user=user, aggregator=agg)
        assert result["orcamind"]["tasks"] == 5
        assert "orcalab" in result
        agg.overview.assert_awaited_once()

    async def test_empty_overview(self, user_factory):
        user = user_factory()
        agg = AsyncMock(spec=Aggregator)
        agg.overview = AsyncMock(return_value={"orcamind": {}, "orcalab": {}, "orcanet": {}})
        result = await overview(_user=user, aggregator=agg)
        assert result == {"orcamind": {}, "orcalab": {}, "orcanet": {}}


class TestPublicStats:
    async def test_returns_stats(self):
        agg = AsyncMock(spec=Aggregator)
        agg.public_stats = AsyncMock(return_value={"tasks": 10, "experiments": 4})
        result = await public_stats(aggregator=agg)
        assert result["tasks"] == 10
        assert result["experiments"] == 4
        agg.public_stats.assert_awaited_once()

    async def test_returns_zero_on_failure(self):
        agg = AsyncMock(spec=Aggregator)
        agg.public_stats = AsyncMock(return_value={"tasks": 0, "experiments": 0})
        result = await public_stats(aggregator=agg)
        assert result["tasks"] == 0
