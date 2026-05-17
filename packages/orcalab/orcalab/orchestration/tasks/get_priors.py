"""Prefect task: fetch model recommendation priors from OrcaMind."""

from __future__ import annotations

import logging
from uuid import UUID

import httpx
from prefect import task

from orca_shared.clients.orcamind_client import OrcaMindClient
from orca_shared.schemas.recommendation import ModelRecommendation, RecommendationRequest

logger = logging.getLogger(__name__)


@task(name="get_orcamind_priors", retries=1)
async def get_orcamind_priors(
    task_id: str, orcamind_url: str
) -> list[ModelRecommendation] | None:
    try:
        async with OrcaMindClient(orcamind_url) as client:
            embedding = await client.embed_task(UUID(task_id))
            req = RecommendationRequest(task_embedding=embedding.embedding_vector)
            recommendation = await client.recommend_model(req)
            return [recommendation]
    except (httpx.ConnectError, httpx.TimeoutException):
        logger.warning("OrcaMind unavailable; starting sweep without priors")
        return None
