from __future__ import annotations

import asyncio
from pathlib import Path

from orca_shared.storage.base import StorageBackend


class LocalStorageBackend(StorageBackend):
    """Filesystem-backed storage for local development and testing."""

    def __init__(self, base_path: str | Path) -> None:
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        path = (self._base / key).resolve()
        if not str(path).startswith(str(self._base.resolve())):
            raise ValueError(f"Key '{key}' escapes the storage root")
        return path

    async def upload(self, key: str, data: bytes) -> str:
        path = self._resolve(key)
        await asyncio.to_thread(_write_file, path, data)
        return str(path)

    async def download(self, key: str) -> bytes:
        path = self._resolve(key)
        return await asyncio.to_thread(_read_file, path)

    async def delete(self, key: str) -> bool:
        path = self._resolve(key)
        return await asyncio.to_thread(_delete_file, path)

    async def list(self, prefix: str = "") -> list[str]:
        def _list() -> list[str]:
            results: list[str] = []
            for p in self._base.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(self._base).as_posix()
                    if rel.startswith(prefix):
                        results.append(rel)
            return sorted(results)

        return await asyncio.to_thread(_list)


def _write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _read_file(path: Path) -> bytes:
    return path.read_bytes()


def _delete_file(path: Path) -> bool:
    if path.exists():
        path.unlink()
        return True
    return False
