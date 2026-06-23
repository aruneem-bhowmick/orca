"""Fan-out aggregator that queries all three upstream services."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from orca_web.config import settings

logger = logging.getLogger("orca_web.aggregator")


class Aggregator:
    """Queries OrcaMind, OrcaLab, and OrcaNet in parallel."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        """Bind the aggregator to an httpx async client for upstream calls."""
        self._client = client

    async def _safe_get(self, url: str) -> dict[str, Any]:
        """GET *url* and return the JSON body, or ``{}`` on any failure."""
        try:
            resp = await self._client.get(url, timeout=10.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("Aggregator request failed (%s): %s", url, exc)
            return {}

    async def overview(self) -> dict[str, Any]:
        """Aggregate high-level stats from all services for the dashboard."""
        tasks_url = f"{settings.orcamind_api_url}/api/v1/tasks?limit=5"
        experiments_url = f"{settings.orcalab_api_url}/api/v1/experiments?limit=5"
        orcamind_health = f"{settings.orcamind_api_url}/health"
        orcalab_health = f"{settings.orcalab_api_url}/health"
        orcanet_health = f"{settings.orcanet_api_url}/health"

        results = await asyncio.gather(
            self._safe_get(tasks_url),
            self._safe_get(experiments_url),
            self._safe_get(orcamind_health),
            self._safe_get(orcalab_health),
            self._safe_get(orcanet_health),
        )

        tasks_data = results[0]
        experiments_data = results[1]

        return {
            "recent_tasks": tasks_data if isinstance(tasks_data, list) else [],
            "recent_experiments": experiments_data if isinstance(experiments_data, list) else [],
            "services": {
                "orcamind": results[2].get("status", "unknown"),
                "orcalab": results[3].get("status", "unknown"),
                "orcanet": results[4].get("status", "unknown"),
            },
        }

    async def public_stats(self) -> dict[str, Any]:
        """Return counts suitable for the public landing page."""
        tasks_url = f"{settings.orcamind_api_url}/api/v1/tasks?limit=1"
        experiments_url = f"{settings.orcalab_api_url}/api/v1/experiments?limit=1"

        tasks_resp, experiments_resp = await asyncio.gather(
            self._safe_get(tasks_url),
            self._safe_get(experiments_url),
        )

        return {
            "task_count": len(tasks_resp) if isinstance(tasks_resp, list) else 0,
            "experiment_count": len(experiments_resp) if isinstance(experiments_resp, list) else 0,
        }
