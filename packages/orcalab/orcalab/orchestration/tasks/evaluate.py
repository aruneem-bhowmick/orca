"""Prefect task: extract requested metrics from an experiment result."""

from __future__ import annotations

from prefect import task

from orca_shared.schemas.training import ExperimentResult


@task(name="evaluate")
async def evaluate(
    result: ExperimentResult,
    metrics: list[str] | None = None,
) -> dict:
    if metrics is None:
        metrics = ["accuracy", "loss"]
    available = result.metrics or {}
    return {m: available.get(m) for m in metrics}
