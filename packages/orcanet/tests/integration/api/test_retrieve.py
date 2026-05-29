"""Integration tests for POST /api/v1/retrieve."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient

from orca_shared.schemas.task import Task


class TestRetrieve:
    async def test_returns_similar_tasks(
        self,
        client: AsyncClient,
        target_task_id: UUID,
        source_task_id: UUID,
    ) -> None:
        response = await client.post(
            "/api/v1/retrieve",
            json={"task_id": str(target_task_id), "top_k": 5},
        )
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) == 1
        assert body[0]["task_id"] == str(source_task_id)
        assert body[0]["rank"] == 1
        assert 0.0 <= body[0]["score"] <= 1.0

    async def test_with_query_description_uses_expanded_retrieval(
        self,
        client: AsyncClient,
        target_task_id: UUID,
        mock_retriever: AsyncMock,
    ) -> None:
        response = await client.post(
            "/api/v1/retrieve",
            json={
                "task_id": str(target_task_id),
                "query_description": "image classification on small data",
                "top_k": 3,
            },
        )
        assert response.status_code == 200
        mock_retriever.retrieve_with_expanded_queries.assert_awaited_once()

    async def test_with_filters_passes_to_retriever(
        self,
        client: AsyncClient,
        target_task_id: UUID,
        mock_retriever: AsyncMock,
    ) -> None:
        response = await client.post(
            "/api/v1/retrieve",
            json={
                "task_id": str(target_task_id),
                "filters": {"domain": "vision"},
            },
        )
        assert response.status_code == 200
        _, kwargs = mock_retriever.retrieve.call_args
        assert kwargs.get("filters") == {"domain": "vision"}

    async def test_empty_results_returns_empty_list(
        self,
        client: AsyncClient,
        target_task_id: UUID,
        mock_retriever: AsyncMock,
    ) -> None:
        mock_retriever.retrieve.return_value = []
        response = await client.post(
            "/api/v1/retrieve",
            json={"task_id": str(target_task_id)},
        )
        assert response.status_code == 200
        assert response.json() == []

    async def test_unknown_task_returns_404(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/retrieve",
            json={"task_id": str(uuid4())},
        )
        assert response.status_code == 404
