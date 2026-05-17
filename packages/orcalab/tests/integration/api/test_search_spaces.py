"""Integration tests for search-space persistence endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient

from orca_shared.schemas.search_space import SearchSpaceRecord


def _search_space_record(search_space_id: UUID | None = None) -> SearchSpaceRecord:
    return SearchSpaceRecord(
        search_space_id=search_space_id or uuid4(),
        name="test_space",
        definition={"name": "test_space", "description": "", "parameters": []},
        created_at=datetime.now(timezone.utc),
    )


class TestCreateSearchSpace:
    async def test_returns_201(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/search-spaces",
            json={"name": "resnet_search", "parameters": []},
        )
        assert resp.status_code == 201

    async def test_response_contains_search_space_id(
        self, client: AsyncClient, search_space_id: UUID
    ) -> None:
        body = (
            await client.post(
                "/api/v1/search-spaces",
                json={"name": "resnet_search"},
            )
        ).json()
        assert UUID(body["search_space_id"]) == search_space_id

    async def test_calls_repo_create(
        self, client: AsyncClient, mock_search_space_repo: AsyncMock
    ) -> None:
        await client.post(
            "/api/v1/search-spaces",
            json={"name": "lr_search", "parameters": [{"type": "float", "name": "lr"}]},
        )
        mock_search_space_repo.create.assert_awaited_once()

    async def test_definition_includes_parameters(
        self, client: AsyncClient, mock_search_space_repo: AsyncMock
    ) -> None:
        params = [{"type": "float", "name": "lr", "low": 1e-5, "high": 1e-1}]
        await client.post(
            "/api/v1/search-spaces",
            json={"name": "test", "parameters": params},
        )
        kwargs = mock_search_space_repo.create.call_args[1]
        assert kwargs["definition"]["parameters"] == params

    async def test_name_passed_to_repo(
        self, client: AsyncClient, mock_search_space_repo: AsyncMock
    ) -> None:
        await client.post(
            "/api/v1/search-spaces", json={"name": "my_space"}
        )
        kwargs = mock_search_space_repo.create.call_args[1]
        assert kwargs["name"] == "my_space"


class TestListSearchSpaces:
    async def test_returns_200(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/search-spaces")
        assert resp.status_code == 200

    async def test_returns_list(
        self, client: AsyncClient, mock_search_space_repo: AsyncMock
    ) -> None:
        mock_search_space_repo.list_all.return_value = []
        body = (await client.get("/api/v1/search-spaces")).json()
        assert isinstance(body, list)

    async def test_calls_list_all(
        self, client: AsyncClient, mock_search_space_repo: AsyncMock
    ) -> None:
        mock_search_space_repo.list_all.return_value = []
        await client.get("/api/v1/search-spaces")
        mock_search_space_repo.list_all.assert_awaited_once()

    async def test_returns_search_space_records(
        self, client: AsyncClient, mock_search_space_repo: AsyncMock
    ) -> None:
        record = _search_space_record()
        mock_search_space_repo.list_all.return_value = [record]
        body = (await client.get("/api/v1/search-spaces")).json()
        assert len(body) == 1
        assert UUID(body[0]["search_space_id"]) == record.search_space_id

    async def test_pagination_params_forwarded(
        self, client: AsyncClient, mock_search_space_repo: AsyncMock
    ) -> None:
        mock_search_space_repo.list_all.return_value = []
        await client.get("/api/v1/search-spaces?limit=5&offset=15")
        kwargs = mock_search_space_repo.list_all.call_args[1]
        assert kwargs["limit"] == 5
        assert kwargs["offset"] == 15
