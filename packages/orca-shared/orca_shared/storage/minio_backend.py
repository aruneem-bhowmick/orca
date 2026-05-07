from __future__ import annotations

import asyncio
import io
import os

from orca_shared.storage.base import StorageBackend


class MinioStorageBackend(StorageBackend):
    """MinIO (S3-compatible) storage backend.

    Env vars: MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY.
    The bucket is derived from the first path segment of each key,
    e.g. key "models/run-1/weights.pt" → bucket "models", object "run-1/weights.pt".
    """

    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        secure: bool = False,
    ) -> None:
        from minio import Minio  # deferred to avoid hard import at module load

        self._client = Minio(
            endpoint or os.environ["MINIO_ENDPOINT"],
            access_key=access_key or os.environ["MINIO_ACCESS_KEY"],
            secret_key=secret_key or os.environ["MINIO_SECRET_KEY"],
            secure=secure,
        )

    @staticmethod
    def _split_key(key: str) -> tuple[str, str]:
        parts = key.split("/", 1)
        if len(parts) < 2 or not parts[1]:
            raise ValueError(f"Key '{key}' must have at least one path segment after the bucket")
        return parts[0], parts[1]

    def _ensure_bucket(self, bucket: str) -> None:
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)

    async def upload(self, key: str, data: bytes) -> str:
        bucket, obj = self._split_key(key)

        def _put() -> str:
            self._ensure_bucket(bucket)
            self._client.put_object(bucket, obj, io.BytesIO(data), length=len(data))
            return f"s3://{bucket}/{obj}"

        return await asyncio.to_thread(_put)

    async def download(self, key: str) -> bytes:
        bucket, obj = self._split_key(key)

        def _get() -> bytes:
            response = self._client.get_object(bucket, obj)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()

        return await asyncio.to_thread(_get)

    async def delete(self, key: str) -> bool:
        bucket, obj = self._split_key(key)

        def _remove() -> bool:
            try:
                self._client.remove_object(bucket, obj)
                return True
            except Exception:
                return False

        return await asyncio.to_thread(_remove)

    async def list(self, prefix: str = "") -> list[str]:
        bucket, obj_prefix = (prefix.split("/", 1) + [""])[:2] if "/" in prefix else (prefix, "")

        def _list() -> list[str]:
            if not self._client.bucket_exists(bucket):
                return []
            objects = self._client.list_objects(bucket, prefix=obj_prefix or None, recursive=True)
            return [f"{bucket}/{o.object_name}" for o in objects]

        return await asyncio.to_thread(_list)
