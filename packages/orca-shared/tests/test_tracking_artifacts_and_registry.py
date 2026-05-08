"""Tests for ArtifactManager and ModelRegistry.

Both modules defer 'import mlflow' and 'import torch' inside their methods.
We inject mocks via sys.modules so those deferred imports resolve to MagicMocks.
"""
from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from orca_shared.storage.base import StorageBackend
from orca_shared.tracking.artifacts import ArtifactManager
from orca_shared.tracking.model_registry import ModelRegistry


# ---------------------------------------------------------------------------
# ArtifactManager
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_storage() -> AsyncMock:
    storage = AsyncMock(spec=StorageBackend)
    storage.upload = AsyncMock(return_value="s3://models/run-1/w.pt")
    storage.download = AsyncMock(return_value=b"\x80\x04\x95FAKE_PICKLE")
    return storage


@pytest.fixture
def manager(mock_storage) -> ArtifactManager:
    return ArtifactManager(mock_storage)


class TestArtifactManagerUploadModel:
    @pytest.mark.asyncio
    async def test_calls_torch_save(self, manager, mock_torch, mock_mlflow):
        model_obj = {"weights": [1, 2, 3]}
        await manager.upload_model(model_obj, "models/run-1/w.pt")
        mock_torch.save.assert_called_once()
        saved_obj, buf = mock_torch.save.call_args[0]
        assert saved_obj is model_obj
        assert isinstance(buf, io.BytesIO)

    @pytest.mark.asyncio
    async def test_uploads_serialised_bytes_to_storage(
        self, manager, mock_torch, mock_mlflow, mock_storage
    ):
        fake_bytes = b"serialised_model_data"
        mock_torch.save.side_effect = lambda obj, buf: buf.write(fake_bytes)

        await manager.upload_model(object(), "models/run-1/w.pt")

        mock_storage.upload.assert_awaited_once()
        key, data = mock_storage.upload.call_args[0]
        assert key == "models/run-1/w.pt"
        assert data == fake_bytes

    @pytest.mark.asyncio
    async def test_logs_model_uri_to_mlflow(self, manager, mock_torch, mock_mlflow):
        await manager.upload_model(object(), "models/run-1/w.pt")
        mock_mlflow.log_param.assert_called_once_with(
            "model_uri", "s3://models/run-1/w.pt"
        )

    @pytest.mark.asyncio
    async def test_returns_storage_uri(self, manager, mock_torch, mock_mlflow):
        uri = await manager.upload_model(object(), "models/run-1/w.pt")
        assert uri == "s3://models/run-1/w.pt"

    @pytest.mark.asyncio
    async def test_torch_save_receives_bytesio_buffer(self, manager, mock_torch, mock_mlflow):
        await manager.upload_model(object(), "models/w.pt")
        _, buf = mock_torch.save.call_args[0]
        assert isinstance(buf, io.BytesIO)


class TestArtifactManagerDownloadModel:
    @pytest.mark.asyncio
    async def test_calls_storage_download(
        self, manager, mock_torch, mock_mlflow, mock_storage
    ):
        await manager.download_model("s3://models/run-1/w.pt")
        mock_storage.download.assert_awaited_once_with("s3://models/run-1/w.pt")

    @pytest.mark.asyncio
    async def test_calls_torch_load_with_downloaded_bytes(
        self, manager, mock_torch, mock_mlflow, mock_storage
    ):
        fake_bytes = b"\x80\x04\x95TEST"
        mock_storage.download.return_value = fake_bytes

        await manager.download_model("s3://models/run-1/w.pt")

        mock_torch.load.assert_called_once()
        buf_arg = mock_torch.load.call_args[0][0]
        assert isinstance(buf_arg, io.BytesIO)
        assert buf_arg.read() == fake_bytes

    @pytest.mark.asyncio
    async def test_torch_load_uses_weights_only_false(
        self, manager, mock_torch, mock_mlflow, mock_storage
    ):
        await manager.download_model("s3://models/run-1/w.pt")
        _, kwargs = mock_torch.load.call_args
        assert kwargs.get("weights_only") is False

    @pytest.mark.asyncio
    async def test_returns_loaded_object(
        self, manager, mock_torch, mock_mlflow, mock_storage
    ):
        sentinel = {"state_dict": {}}
        mock_torch.load.return_value = sentinel

        result = await manager.download_model("s3://models/run-1/w.pt")
        assert result is sentinel


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------


class TestModelRegistryInit:
    def test_sets_tracking_uri_when_provided(self, mock_mlflow):
        ModelRegistry(tracking_uri="http://mlflow:5000")
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://mlflow:5000")

    def test_skips_tracking_uri_when_none(self, mock_mlflow):
        ModelRegistry()
        mock_mlflow.set_tracking_uri.assert_not_called()

    def test_creates_mlflow_client(self, mock_mlflow):
        reg = ModelRegistry()
        mock_mlflow.MlflowClient.assert_called_once()
        assert reg._client is mock_mlflow.MlflowClient.return_value


class TestModelRegistryRegister:
    def test_constructs_runs_uri(self, mock_mlflow):
        reg = ModelRegistry()
        reg.register("run-123", "model/artifacts", "MyModel")
        mock_mlflow.register_model.assert_called_once_with(
            model_uri="runs:/run-123/model/artifacts", name="MyModel"
        )

    def test_returns_registered_model_version(self, mock_mlflow):
        mock_version = MagicMock(version="3")
        mock_mlflow.register_model.return_value = mock_version

        reg = ModelRegistry()
        result = reg.register("run-xyz", "model", "Net")
        assert result.version == "3"

    def test_different_run_ids(self, mock_mlflow):
        reg = ModelRegistry()
        reg.register("run-A", "path", "ModelA")
        uri_used = mock_mlflow.register_model.call_args[1]["model_uri"]
        assert "run-A" in uri_used


class TestModelRegistryTransitionStage:
    def test_delegates_to_mlflow_client(self, mock_mlflow):
        reg = ModelRegistry()
        reg.transition_stage("MyModel", "2", "Production")
        reg._client.transition_model_version_stage.assert_called_once_with(
            name="MyModel", version="2", stage="Production"
        )

    def test_returns_result(self, mock_mlflow):
        mock_result = MagicMock()
        reg = ModelRegistry()
        reg._client.transition_model_version_stage.return_value = mock_result
        result = reg.transition_stage("M", "1", "Staging")
        assert result is mock_result


class TestModelRegistryGetLatest:
    def test_passes_stage_as_list(self, mock_mlflow):
        reg = ModelRegistry()
        reg.get_latest("MyModel", stage="Staging")
        reg._client.get_latest_versions.assert_called_once_with(
            "MyModel", stages=["Staging"]
        )

    def test_passes_empty_list_when_stage_none(self, mock_mlflow):
        reg = ModelRegistry()
        reg.get_latest("MyModel")
        reg._client.get_latest_versions.assert_called_once_with("MyModel", stages=[])

    def test_returns_versions_list(self, mock_mlflow):
        mock_versions = [MagicMock(version="1"), MagicMock(version="2")]
        reg = ModelRegistry()
        reg._client.get_latest_versions.return_value = mock_versions
        result = reg.get_latest("MyModel")
        assert result is mock_versions
