from __future__ import annotations

from typing import Any


class MetricLogger:
    """Thin wrapper around mlflow metric logging with optional run scoping."""

    def __init__(self, run_id: str | None = None) -> None:
        self._run_id = run_id

    def log(self, name: str, value: float, step: int | None = None) -> None:
        import mlflow

        mlflow.log_metric(name, value, step=step, run_id=self._run_id)

    def log_batch(self, metrics: dict[str, float], step: int | None = None) -> None:
        for name, value in metrics.items():
            self.log(name, value, step=step)
