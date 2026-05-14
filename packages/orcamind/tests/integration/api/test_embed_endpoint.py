"""Integration tests for POST /api/v1/tasks/embed — Embedding schema validation.

These tests complement TestEmbedTask in test_tasks.py (which covers HTTP status
codes and side-effect calls). The focus here is exhaustive validation of every
field in the Embedding response schema and dimension variants.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from pydantic import ValidationError

from orca_shared.schemas.embedding import Embedding


def _embed_body(
    task_id: UUID,
    vector: list[float],
    embedding_type: str = "statistical",
    model_version: str = "v1",
) -> dict:
    return {
        "task_id": str(task_id),
        "embedding_vector": vector,
        "embedding_type": embedding_type,
        "model_version": model_version,
    }


class TestEmbeddingSchemaFields:
    """Verify every field of the Embedding response schema is present and correctly typed."""

    async def test_response_contains_all_seven_fields(
        self, client: AsyncClient, task_id: UUID
    ) -> None:
        body = (
            await client.post(
                "/api/v1/tasks/embed", json=_embed_body(task_id, [0.0] * 25)
            )
        ).json()
        expected = {
            "embedding_id",
            "task_id",
            "embedding_type",
            "embedding_vector",
            "dimension",
            "model_version",
            "created_at",
        }
        assert expected.issubset(body.keys())

    async def test_embedding_id_is_valid_uuid(
        self, client: AsyncClient, task_id: UUID
    ) -> None:
        body = (
            await client.post(
                "/api/v1/tasks/embed", json=_embed_body(task_id, [0.0] * 25)
            )
        ).json()
        UUID(body["embedding_id"])  # raises ValueError if malformed

    async def test_task_id_in_response_matches_request(
        self, client: AsyncClient, task_id: UUID
    ) -> None:
        body = (
            await client.post(
                "/api/v1/tasks/embed", json=_embed_body(task_id, [0.0] * 25)
            )
        ).json()
        assert UUID(body["task_id"]) == task_id

    async def test_embedding_type_reflects_request(
        self, client: AsyncClient, task_id: UUID
    ) -> None:
        body = (
            await client.post(
                "/api/v1/tasks/embed",
                json=_embed_body(task_id, [0.0] * 25, embedding_type="statistical"),
            )
        ).json()
        assert body["embedding_type"] == "statistical"

    async def test_embedding_vector_is_list_of_numbers(
        self, client: AsyncClient, task_id: UUID
    ) -> None:
        body = (
            await client.post(
                "/api/v1/tasks/embed", json=_embed_body(task_id, [0.1] * 25)
            )
        ).json()
        assert isinstance(body["embedding_vector"], list)
        assert all(isinstance(v, (int, float)) for v in body["embedding_vector"])

    async def test_dimension_matches_vector_length(
        self, client: AsyncClient, task_id: UUID
    ) -> None:
        body = (
            await client.post(
                "/api/v1/tasks/embed", json=_embed_body(task_id, [0.0] * 25)
            )
        ).json()
        assert body["dimension"] == len(body["embedding_vector"])

    async def test_model_version_reflects_request(
        self, client: AsyncClient, task_id: UUID
    ) -> None:
        body = (
            await client.post(
                "/api/v1/tasks/embed",
                json=_embed_body(task_id, [0.0] * 25, model_version="v1"),
            )
        ).json()
        assert body["model_version"] == "v1"

    async def test_created_at_is_parseable_iso_datetime(
        self, client: AsyncClient, task_id: UUID
    ) -> None:
        body = (
            await client.post(
                "/api/v1/tasks/embed", json=_embed_body(task_id, [0.0] * 25)
            )
        ).json()
        # datetime.fromisoformat raises ValueError if the string is malformed
        datetime.fromisoformat(body["created_at"])


class TestEmbeddingDimensions:
    """Verify dimension variants and the Embedding schema's consistency validator."""

    async def test_statistical_embedding_dimension_25(
        self, client: AsyncClient, task_id: UUID
    ) -> None:
        body = (
            await client.post(
                "/api/v1/tasks/embed",
                json=_embed_body(task_id, [0.1] * 25, embedding_type="statistical"),
            )
        ).json()
        assert body["dimension"] == 25
        assert len(body["embedding_vector"]) == 25

    async def test_neural_embedding_dimension_64(
        self,
        client: AsyncClient,
        task_id: UUID,
        embedding_id: UUID,
        mock_embedding_repo: AsyncMock,
        now: datetime,
    ) -> None:
        mock_embedding_repo.create.return_value = Embedding(
            embedding_id=embedding_id,
            task_id=task_id,
            embedding_type="neural",
            embedding_vector=[0.0] * 64,
            dimension=64,
            model_version="v2",
            created_at=now,
        )
        body = (
            await client.post(
                "/api/v1/tasks/embed",
                json=_embed_body(task_id, [0.0] * 64, embedding_type="neural", model_version="v2"),
            )
        ).json()
        assert body["dimension"] == 64
        assert body["embedding_type"] == "neural"
        assert len(body["embedding_vector"]) == 64

    def test_schema_rejects_dimension_vector_mismatch(
        self, task_id: UUID, embedding_id: UUID, now: datetime
    ) -> None:
        with pytest.raises(ValidationError):
            Embedding(
                embedding_id=embedding_id,
                task_id=task_id,
                embedding_type="statistical",
                embedding_vector=[0.0] * 25,
                dimension=99,  # intentionally wrong
                model_version="v1",
                created_at=now,
            )
