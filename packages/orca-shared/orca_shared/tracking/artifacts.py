from __future__ import annotations

import io
import tempfile
from pathlib import Path

from orca_shared.storage.base import StorageBackend


class ArtifactManager:
    """Manages model serialisation, remote storage, and MLflow URI tracking."""

    def __init__(self, storage: StorageBackend) -> None:
        self._storage = storage

    async def upload_model(self, model: object, name: str) -> str:
        """Serialise *model* with torch.save, upload via storage, log URI to MLflow."""
        import mlflow
        import torch

        buf = io.BytesIO()
        torch.save(model, buf)
        uri = await self._storage.upload(name, buf.getvalue())
        mlflow.log_param("model_uri", uri)
        return uri

    async def download_model(self, uri: str) -> object:
        """Download bytes from *uri* and deserialise with torch.load."""
        import torch

        data = await self._storage.download(uri)
        return torch.load(io.BytesIO(data), weights_only=True)
