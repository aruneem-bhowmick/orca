from __future__ import annotations

from typing import Any
from uuid import UUID

from orca_shared.clients._base import _BaseAsyncClient
from orca_shared.schemas.transfer import TransferRecommendation, TransferScore


class OrcaNetClient(_BaseAsyncClient):
    """Async httpx client for the OrcaNet transfer learning service."""

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
