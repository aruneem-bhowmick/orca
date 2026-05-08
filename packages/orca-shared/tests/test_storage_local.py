"""Roundtrip tests for LocalStorageBackend (upload/download/delete/list)."""
from __future__ import annotations

from pathlib import Path

import pytest

from orca_shared.storage.local_backend import LocalStorageBackend


@pytest.mark.asyncio
async def test_upload_creates_file(storage_base_path: Path):
    backend = LocalStorageBackend(storage_base_path)
    uri = await backend.upload("test/hello.bin", b"hello")
    assert Path(uri).exists()
    assert Path(uri).read_bytes() == b"hello"


@pytest.mark.asyncio
async def test_download_roundtrip(storage_base_path: Path):
    backend = LocalStorageBackend(storage_base_path)
    payload = b"\x00\x01\x02\x03" * 100
    await backend.upload("data/chunk.bin", payload)
    result = await backend.download("data/chunk.bin")
    assert result == payload


@pytest.mark.asyncio
async def test_delete_removes_file(storage_base_path: Path):
    backend = LocalStorageBackend(storage_base_path)
    await backend.upload("tmp/remove_me.txt", b"bye")
    deleted = await backend.delete("tmp/remove_me.txt")
    assert deleted is True
    assert not (storage_base_path / "tmp" / "remove_me.txt").exists()


@pytest.mark.asyncio
async def test_delete_nonexistent_returns_false(storage_base_path: Path):
    backend = LocalStorageBackend(storage_base_path)
    deleted = await backend.delete("ghost/file.bin")
    assert deleted is False


@pytest.mark.asyncio
async def test_list_returns_keys(storage_base_path: Path):
    backend = LocalStorageBackend(storage_base_path)
    await backend.upload("models/a.pt", b"a")
    await backend.upload("models/b.pt", b"b")
    await backend.upload("data/c.csv", b"c")
    keys = await backend.list()
    assert "models/a.pt" in keys
    assert "models/b.pt" in keys
    assert "data/c.csv" in keys


@pytest.mark.asyncio
async def test_list_with_prefix_filters(storage_base_path: Path):
    backend = LocalStorageBackend(storage_base_path)
    await backend.upload("runs/1/weights.pt", b"w1")
    await backend.upload("runs/2/weights.pt", b"w2")
    await backend.upload("config/settings.yaml", b"cfg")
    keys = await backend.list(prefix="runs/")
    assert all(k.startswith("runs/") for k in keys)
    assert "config/settings.yaml" not in keys


@pytest.mark.asyncio
async def test_path_traversal_rejected(storage_base_path: Path):
    backend = LocalStorageBackend(storage_base_path)
    with pytest.raises(ValueError):
        await backend.upload("../../etc/passwd", b"evil")
