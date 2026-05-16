"""Prefect task: log experiment results to OrcaMind."""

from __future__ import annotations

import logging

from prefect import task

from orca_shared.clients.orcamind_client import OrcaMindClient
from orca_shared.schemas.recommendation import FeedbackRequest
from orca_shared.schemas.training import ExperimentResult

logger = logging.getLogger(__name__)


@task(name="log_results")
async def log_results(result: ExperimentResult, orcamind_client: OrcaMindClient) -> None:
    req = FeedbackRequest(
        experiment_id=result.experiment_id,
        actual_metric=max(result.metrics.values()) if result.metrics else 0.0,
        metric_name="objective",
    )
    try:
        await orcamind_client.submit_feedback(req)
    except NotImplementedError:
        logger.debug("OrcaMind client not yet implemented; skipping result logging")
