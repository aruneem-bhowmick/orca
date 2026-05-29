"""Integration tests for POST /api/v1/cross-domain-embed."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
import torch
from httpx import AsyncClient


class TestCrossDomainEmbed:
    async def test_embed_by_task_id(
        self,
        client: AsyncClient,
        target_task_id: UUID,
    ) -> None:
        response = await client.post(
            "/api/v1/cross-domain-embed",
            json={"task_id": str(target_task_id)},
        )
        assert response.status_code == 200
        body = response.json()
        assert "embedding" in body
        assert len(body["embedding"]) == 64

    async def test_embed_by_statistical_features(
        self,
        client: AsyncClient,
    ) -> None:
        features = [float(i) for i in range(25)]
        response = await client.post(
            "/api/v1/cross-domain-embed",
            json={"statistical_features": features},
        )
        assert response.status_code == 200
        body = response.json()
        assert "embedding" in body
        assert len(body["embedding"]) == 64

    async def test_missing_both_fields_returns_422(
        self,
        client: AsyncClient,
    ) -> None:
        response = await client.post(
            "/api/v1/cross-domain-embed",
            json={"description": "only description, no task id or features"},
        )
        assert response.status_code == 422

    async def test_unknown_task_id_returns_404(
        self,
        client: AsyncClient,
    ) -> None:
        response = await client.post(
            "/api/v1/cross-domain-embed",
            json={"task_id": str(uuid4())},
        )
        assert response.status_code == 404

    async def test_embedding_values_come_from_embedder(
        self,
        client: AsyncClient,
        target_task_id: UUID,
        mock_embedder: MagicMock,
    ) -> None:
        sentinel = torch.ones(1, 64) * 0.5
        mock_embedder.embed.return_value = sentinel

        response = await client.post(
            "/api/v1/cross-domain-embed",
            json={"task_id": str(target_task_id)},
        )
        assert response.status_code == 200
        body = response.json()
        assert all(pytest.approx(v) == 0.5 for v in body["embedding"])
