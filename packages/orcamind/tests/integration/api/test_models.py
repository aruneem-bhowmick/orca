"""Integration tests for GET /api/v1/models."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from httpx import AsyncClient

from orca_shared.registry.models import Model as ModelORM


def _make_model_row() -> MagicMock:
    row = MagicMock(spec=ModelORM)
    row.model_id = uuid4()
    row.name = "resnet18"
    row.architecture = "cnn"
    row.config = {"depth": 18}
    row.parameter_count = 11_000_000
    row.flops = None
    return row


class TestListModels:
    async def test_returns_200(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/models")
        assert resp.status_code == 200

    async def test_returns_list(self, client: AsyncClient) -> None:
        body = (await client.get("/api/v1/models")).json()
        assert isinstance(body, list)

    async def test_empty_list_when_no_models(
        self, client: AsyncClient, mock_session: AsyncMock
    ) -> None:
        scalars_mock = MagicMock()
        scalars_mock.scalars.return_value = iter([])
        mock_session.execute.return_value = scalars_mock

        body = (await client.get("/api/v1/models")).json()
        assert body == []

    async def test_returns_model_configs(
        self, client: AsyncClient, mock_session: AsyncMock
    ) -> None:
        row = _make_model_row()
        scalars_mock = MagicMock()
        scalars_mock.scalars.return_value = iter([row])
        mock_session.execute.return_value = scalars_mock

        body = (await client.get("/api/v1/models")).json()
        assert len(body) == 1
        assert body[0]["name"] == "resnet18"
        assert body[0]["architecture"] == "cnn"

    async def test_limit_query_param_accepted(
        self, client: AsyncClient
    ) -> None:
        resp = await client.get("/api/v1/models?limit=10")
        assert resp.status_code == 200
