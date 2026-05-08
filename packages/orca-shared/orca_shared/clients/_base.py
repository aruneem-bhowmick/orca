from __future__ import annotations

import httpx


class _BaseAsyncClient:
    """Shared async httpx lifecycle for inter-service clients."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        max_connections: int = 20,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=max_connections),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self: "_BaseAsyncClient") -> "_BaseAsyncClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
