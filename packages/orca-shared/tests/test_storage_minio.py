"""Tests for MinioStorageBackend.

minio.Minio is injected via sys.modules so the deferred 'from minio import Minio'
inside __init__ resolves to our mock. No real MinIO server needed.
"""
from __future__ import annotations

import io
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from orca_shared.storage.minio_backend import MinioStorageBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_minio_module(monkeypatch):
    """Replace the entire minio package with a mock before each test."""
    mock_mod = MagicMock()
    monkeypatch.setitem(sys.modules, "minio", mock_mod)
    return mock_mod


@pytest.fixture
def mock_client(mock_minio_module):
    """The Minio instance that will be created inside the backend constructor."""
    client = MagicMock()
    mock_minio_module.Minio.return_value = client
    return client


@pytest.fixture
def backend(mock_client):
    return MinioStorageBackend("localhost:9000", "access", "secret")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestMinioInit:
    def test_explicit_params_passed_to_minio(self, mock_minio_module):
        mock_minio_module.Minio.return_value = MagicMock()
        MinioStorageBackend("host:9000", "key", "sec")
        mock_minio_module.Minio.assert_called_once_with(
            "host:9000", access_key="key", secret_key="sec", secure=False
        )

    def test_secure_flag_forwarded(self, mock_minio_module):
        mock_minio_module.Minio.return_value = MagicMock()
        MinioStorageBackend("host:9000", "key", "sec", secure=True)
        _, kwargs = mock_minio_module.Minio.call_args
        assert kwargs["secure"] is True

    def test_reads_endpoint_from_env(self, mock_minio_module, monkeypatch):
        mock_minio_module.Minio.return_value = MagicMock()
        monkeypatch.setenv("MINIO_ENDPOINT", "env-host:9000")
        monkeypatch.setenv("MINIO_ACCESS_KEY", "env-key")
        monkeypatch.setenv("MINIO_SECRET_KEY", "env-secret")
        MinioStorageBackend()
        args, _ = mock_minio_module.Minio.call_args
        assert args[0] == "env-host:9000"

    def test_reads_credentials_from_env(self, mock_minio_module, monkeypatch):
        mock_minio_module.Minio.return_value = MagicMock()
        monkeypatch.setenv("MINIO_ENDPOINT", "env-host:9000")
        monkeypatch.setenv("MINIO_ACCESS_KEY", "env-key")
        monkeypatch.setenv("MINIO_SECRET_KEY", "env-secret")
        MinioStorageBackend()
        _, kwargs = mock_minio_module.Minio.call_args
        assert kwargs["access_key"] == "env-key"
        assert kwargs["secret_key"] == "env-secret"

    def test_missing_endpoint_env_raises(self, mock_minio_module, monkeypatch):
        mock_minio_module.Minio.return_value = MagicMock()
        monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
        monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
        monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)
        with pytest.raises(KeyError):
            MinioStorageBackend()


# ---------------------------------------------------------------------------
# _split_key
# ---------------------------------------------------------------------------


class TestSplitKey:
    def test_valid_two_segment_key(self):
        bucket, obj = MinioStorageBackend._split_key("models/weights.pt")
        assert bucket == "models"
        assert obj == "weights.pt"

    def test_valid_nested_key(self):
        bucket, obj = MinioStorageBackend._split_key("models/run-1/epoch-5/weights.pt")
        assert bucket == "models"
        assert obj == "run-1/epoch-5/weights.pt"

    def test_key_without_slash_raises(self):
        with pytest.raises(ValueError, match="at least one path segment"):
            MinioStorageBackend._split_key("noslash")

    def test_key_with_trailing_slash_only_raises(self):
        with pytest.raises(ValueError, match="at least one path segment"):
            MinioStorageBackend._split_key("bucket/")


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------


class TestMinioUpload:
    @pytest.mark.asyncio
    async def test_creates_bucket_when_missing(self, backend, mock_client):
        mock_client.bucket_exists.return_value = False
        await backend.upload("models/w.pt", b"data")
        mock_client.make_bucket.assert_called_once_with("models")

    @pytest.mark.asyncio
    async def test_skips_make_bucket_when_exists(self, backend, mock_client):
        mock_client.bucket_exists.return_value = True
        await backend.upload("models/w.pt", b"data")
        mock_client.make_bucket.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_put_object_with_correct_args(self, backend, mock_client):
        mock_client.bucket_exists.return_value = True
        payload = b"hello world"
        await backend.upload("models/w.pt", payload)
        mock_client.put_object.assert_called_once()
        args, kwargs = mock_client.put_object.call_args
        bucket, obj, stream, *_ = args
        assert bucket == "models"
        assert obj == "w.pt"
        assert isinstance(stream, io.BytesIO)
        assert kwargs.get("length") == len(payload) or args[3] == len(payload)

    @pytest.mark.asyncio
    async def test_returns_s3_uri(self, backend, mock_client):
        mock_client.bucket_exists.return_value = True
        uri = await backend.upload("models/run-1/w.pt", b"x")
        assert uri == "s3://models/run-1/w.pt"

    @pytest.mark.asyncio
    async def test_upload_empty_bytes(self, backend, mock_client):
        mock_client.bucket_exists.return_value = True
        uri = await backend.upload("data/empty.bin", b"")
        assert uri == "s3://data/empty.bin"


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


class TestMinioDownload:
    @pytest.mark.asyncio
    async def test_calls_get_object(self, backend, mock_client):
        mock_response = MagicMock()
        mock_response.read.return_value = b"payload"
        mock_client.get_object.return_value = mock_response

        result = await backend.download("models/w.pt")

        mock_client.get_object.assert_called_once_with("models", "w.pt")
        assert result == b"payload"

    @pytest.mark.asyncio
    async def test_response_closed_after_read(self, backend, mock_client):
        mock_response = MagicMock()
        mock_response.read.return_value = b"data"
        mock_client.get_object.return_value = mock_response

        await backend.download("models/w.pt")

        mock_response.close.assert_called_once()
        mock_response.release_conn.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_bytes(self, backend, mock_client):
        mock_response = MagicMock()
        mock_response.read.return_value = b"\x00\x01\x02"
        mock_client.get_object.return_value = mock_response

        result = await backend.download("data/chunk.bin")
        assert isinstance(result, bytes)
        assert result == b"\x00\x01\x02"


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestMinioDelete:
    @pytest.mark.asyncio
    async def test_calls_remove_object(self, backend, mock_client):
        await backend.delete("models/w.pt")
        mock_client.remove_object.assert_called_once_with("models", "w.pt")

    @pytest.mark.asyncio
    async def test_returns_true_on_success(self, backend, mock_client):
        result = await backend.delete("models/w.pt")
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_exception(self, backend, mock_client):
        mock_client.remove_object.side_effect = Exception("NoSuchKey")
        result = await backend.delete("models/w.pt")
        assert result is False


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestMinioList:
    @pytest.mark.asyncio
    async def test_calls_list_objects_with_bucket_and_prefix(self, backend, mock_client):
        mock_client.bucket_exists.return_value = True
        mock_client.list_objects.return_value = []
        await backend.list("models/run-1/")
        mock_client.list_objects.assert_called_once_with(
            "models", prefix="run-1/", recursive=True
        )

    @pytest.mark.asyncio
    async def test_returns_formatted_keys(self, backend, mock_client):
        mock_client.bucket_exists.return_value = True
        obj1 = SimpleNamespace(object_name="run-1/w.pt")
        obj2 = SimpleNamespace(object_name="run-2/w.pt")
        mock_client.list_objects.return_value = [obj1, obj2]

        keys = await backend.list("models/")
        assert keys == ["models/run-1/w.pt", "models/run-2/w.pt"]

    @pytest.mark.asyncio
    async def test_returns_empty_list_if_bucket_not_found(self, backend, mock_client):
        mock_client.bucket_exists.return_value = False
        keys = await backend.list("nonexistent/")
        assert keys == []
        mock_client.list_objects.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_prefix_passes_none_to_list_objects(self, backend, mock_client):
        mock_client.bucket_exists.return_value = True
        mock_client.list_objects.return_value = []
        # When prefix has no slash, bucket is the prefix and obj_prefix is empty
        await backend.list("mybucket")
        mock_client.list_objects.assert_called_once_with(
            "mybucket", prefix=None, recursive=True
        )
