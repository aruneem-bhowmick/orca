from __future__ import annotations

from typing import Any


class ModelRegistry:
    """Wraps the MLflow model registry for version management."""

    def __init__(self, tracking_uri: str | None = None) -> None:
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self._client = mlflow.MlflowClient()

    def register(self, run_id: str, model_path: str, name: str) -> Any:
        """Register a logged model artifact as a new version under *name*."""
        import mlflow

        model_uri = f"runs:/{run_id}/{model_path}"
        return mlflow.register_model(model_uri=model_uri, name=name)

    def transition_stage(self, name: str, version: str, stage: str) -> Any:
        """Move *version* of model *name* to the given *stage*."""
        return self._client.transition_model_version_stage(
            name=name, version=version, stage=stage
        )

    def get_latest(self, name: str, stage: str | None = None) -> list[Any]:
        """Return latest versions of *name*, optionally filtered by *stage*."""
        stages = [stage] if stage else []
        return self._client.get_latest_versions(name, stages=stages)
