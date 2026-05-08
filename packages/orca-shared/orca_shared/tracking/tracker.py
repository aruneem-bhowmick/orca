from __future__ import annotations

from types import TracebackType
from typing import Any


class OrcaTracker:
    """Async context manager that wraps an MLflow run lifetime."""

    def __init__(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        self._experiment_name = experiment_name
        self._run_name = run_name
        self._tracking_uri = tracking_uri
        self._run = None

    async def __aenter__(self) -> "OrcaTracker":
        import mlflow

        if self._tracking_uri:
            mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._experiment_name)
        self._run = mlflow.start_run(run_name=self._run_name)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        import mlflow

        status = "FAILED" if exc_type is not None else "FINISHED"
        mlflow.end_run(status=status)

    def log_params(self, params: dict[str, Any]) -> None:
        import mlflow

        mlflow.log_params(params)

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        import mlflow

        mlflow.log_metric(name, value, step=step)

    def log_artifact(self, local_path: str) -> None:
        import mlflow

        mlflow.log_artifact(local_path)
