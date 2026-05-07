from __future__ import annotations

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Abstract base class for all Orca object storage backends."""

    @abstractmethod
    async def upload(self, key: str, data: bytes) -> str:
        """Persist *data* under *key* and return the storage URI."""

    @abstractmethod
    async def download(self, key: str) -> bytes:
        """Retrieve and return the raw bytes stored at *key*."""

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Remove the object at *key*. Return True if it existed."""

    @abstractmethod
    async def list(self, prefix: str = "") -> list[str]:
        """Return all keys that start with *prefix*."""
