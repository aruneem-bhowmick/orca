from __future__ import annotations

from typing import Any
from uuid import UUID

import httpx

from orca_shared.schemas.transfer import TransferRecommendation, TransferScore


class OrcaNetClient:
    """Async httpx client for the OrcaNet transfer learning service."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=20),
        )

    async def score_transfer(
        self, source_id: UUID, target_id: UUID
    ) -> TransferScore:
        raise NotImplementedError("OrcaNetClient.score_transfer is not yet implemented")

    async def recommend_transfer(
        self, target_id: UUID, top_k: int = 5
    ) -> TransferRecommendation:
        raise NotImplementedError("OrcaNetClient.recommend_transfer is not yet implemented")

    async def explain_transfer(self, mapping_id: UUID) -> dict[str, Any]:
        raise NotImplementedError("OrcaNetClient.explain_transfer is not yet implemented")

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OrcaNetClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
