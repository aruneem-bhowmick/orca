"""Integration tests for GET /api/v1/tasks, GET /api/v1/tasks/{id}, POST /api/v1/tasks/embed."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient

from orca_shared.schemas.task import Task, TaskSummary


class TestListTasks:
    async def test_returns_200(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/tasks")
        assert resp.status_code == 200

    async def test_returns_list(
        self, client: AsyncClient, mock_task_repo: AsyncMock
    ) -> None:
        mock_task_repo.list_all.return_value = []
        body = (await client.get("/api/v1/tasks")).json()
        assert isinstance(body, list)

    async def test_domain_filter_calls_list_by_domain(
        self, client: AsyncClient, mock_task_repo: AsyncMock
    ) -> None:
        mock_task_repo.list_by_domain.return_value = []
        await client.get("/api/v1/tasks?domain=tabular")
        mock_task_repo.list_by_domain.assert_awaited_once()
        args = mock_task_repo.list_by_domain.call_args
        assert args[0][0] == "tabular"

    async def test_task_type_filter_calls_list_by_type(
        self, client: AsyncClient, mock_task_repo: AsyncMock
    ) -> None:
        mock_task_repo.list_by_type.return_value = []
        await client.get("/api/v1/tasks?task_type=classification")
        mock_task_repo.list_by_type.assert_awaited_once()
        args = mock_task_repo.list_by_type.call_args
        assert args[0][0] == "classification"

    async def test_no_filter_calls_list_all(
        self, client: AsyncClient, mock_task_repo: AsyncMock
    ) -> None:
        mock_task_repo.list_all.return_value = []
        await client.get("/api/v1/tasks")
        mock_task_repo.list_all.assert_awaited_once()

    async def test_limit_and_offset_forwarded(
        self, client: AsyncClient, mock_task_repo: AsyncMock
    ) -> None:
        mock_task_repo.list_all.return_value = []
        await client.get("/api/v1/tasks?limit=10&offset=20")
        kwargs = mock_task_repo.list_all.call_args[1]
        assert kwargs["limit"] == 10
        assert kwargs["offset"] == 20

    async def test_returns_422_when_both_domain_and_task_type_provided(
        self, client: AsyncClient
    ) -> None:
        resp = await client.get("/api/v1/tasks?domain=tabular&task_type=classification")
        assert resp.status_code == 422


class TestGetTask:
    async def test_returns_200_when_found(
        self, client: AsyncClient, task_id: UUID
    ) -> None:
        resp = await client.get(f"/api/v1/tasks/{task_id}")
        assert resp.status_code == 200

    async def test_response_contains_task_id(
        self, client: AsyncClient, task_id: UUID
    ) -> None:
        body = (await client.get(f"/api/v1/tasks/{task_id}")).json()
        assert UUID(body["task_id"]) == task_id

    async def test_returns_404_when_not_found(
        self, client: AsyncClient, mock_task_repo: AsyncMock
    ) -> None:
        mock_task_repo.get_by_id.return_value = None
        resp = await client.get(f"/api/v1/tasks/{uuid4()}")
        assert resp.status_code == 404

    async def test_returns_422_on_non_uuid(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/tasks/not-a-uuid")
        assert resp.status_code == 422


class TestEmbedTask:
    def _valid_body(self, task_id: UUID) -> dict:
        return {
            "task_id": str(task_id),
            "embedding_vector": [0.1] * 25,
            "embedding_type": "statistical",
            "model_version": "v1",
        }

    async def test_returns_200(
        self, client: AsyncClient, task_id: UUID
    ) -> None:
        resp = await client.post("/api/v1/tasks/embed", json=self._valid_body(task_id))
        assert resp.status_code == 200

    async def test_response_has_embedding_fields(
        self, client: AsyncClient, task_id: UUID
    ) -> None:
        body = (
            await client.post("/api/v1/tasks/embed", json=self._valid_body(task_id))
        ).json()
        assert "embedding_id" in body
        assert "embedding_vector" in body
        assert body["dimension"] == 25

    async def test_calls_emb_repo_create(
        self,
        client: AsyncClient,
        task_id: UUID,
        mock_embedding_repo: AsyncMock,
    ) -> None:
        await client.post("/api/v1/tasks/embed", json=self._valid_body(task_id))
        mock_embedding_repo.create.assert_awaited_once()

    async def test_calls_update_embedding(
        self,
        client: AsyncClient,
        task_id: UUID,
        mock_task_repo: AsyncMock,
    ) -> None:
        await client.post("/api/v1/tasks/embed", json=self._valid_body(task_id))
        mock_task_repo.update_embedding.assert_awaited_once()

    async def test_returns_404_when_task_missing(
        self, client: AsyncClient, mock_task_repo: AsyncMock
    ) -> None:
        mock_task_repo.get_by_id.return_value = None
        resp = await client.post(
            "/api/v1/tasks/embed",
            json=self._valid_body(uuid4()),
        )
        assert resp.status_code == 404

    async def test_returns_422_on_missing_body(
        self, client: AsyncClient
    ) -> None:
        resp = await client.post("/api/v1/tasks/embed", json={})
        assert resp.status_code == 422
