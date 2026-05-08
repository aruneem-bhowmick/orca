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

    def set_alias(self, name: str, version: str, alias: str) -> None:
        """Assign *alias* to *version* of model *name* (MLflow 2.10+ API)."""
        self._client.set_registered_model_alias(name=name, alias=alias, version=version)

    def get_by_alias(self, name: str, alias: str) -> Any:
        """Return the model version identified by *alias* (MLflow 2.10+ API)."""
        return self._client.get_model_version_by_alias(name=name, alias=alias)

    def transition_stage(self, name: str, version: str, stage: str) -> Any:
        """Legacy shim: move *version* to *stage*; prefer set_alias on MLflow >=2.10."""
        self.set_alias(name, version, stage)

    def get_latest(self, name: str, stage: str | None = None) -> list[Any]:
        """Legacy shim: return latest versions; prefer get_by_alias on MLflow >=2.10."""
        if stage:
            return [self.get_by_alias(name, stage)]
        stages: list[str] = []
        return self._client.get_latest_versions(name, stages=stages)
