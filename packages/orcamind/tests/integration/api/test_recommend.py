"""Integration tests for POST /api/v1/recommend-model, predict-performance, similar-tasks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient

from orca_shared.schemas.recommendation import ModelRecommendation


_EMBEDDING = [0.1] * 25


class TestRecommendModel:
    def _body(self, top_k: int = 3) -> dict:
        return {"task_embedding": _EMBEDDING, "top_k": top_k}

    async def test_returns_200(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/recommend-model", json=self._body())
        assert resp.status_code == 200

    async def test_returns_list(self, client: AsyncClient) -> None:
        body = (await client.post("/api/v1/recommend-model", json=self._body())).json()
        assert isinstance(body, list)

    async def test_calls_selector_recommend(
        self, client: AsyncClient, mock_nn_selector: MagicMock
    ) -> None:
        await client.post("/api/v1/recommend-model", json=self._body(top_k=2))
        mock_nn_selector.recommend.assert_called_once()
        _, kwargs = mock_nn_selector.recommend.call_args
        assert kwargs.get("top_k") == 2

    async def test_returns_503_when_selector_not_fitted(
        self, client: AsyncClient, mock_nn_selector: MagicMock
    ) -> None:
        mock_nn_selector.recommend.side_effect = RuntimeError("Selector has not been fitted yet.")
        resp = await client.post("/api/v1/recommend-model", json=self._body())
        assert resp.status_code == 503

    async def test_returns_422_on_bad_body(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/recommend-model", json={})
        assert resp.status_code == 422


class TestPredictPerformance:
    def _body(self, model_id: UUID) -> dict:
        return {"task_embedding": _EMBEDDING, "model_id": str(model_id)}

    async def test_returns_200(
        self,
        client: AsyncClient,
        model_id: UUID,
        mock_session: AsyncMock,
    ) -> None:
        from unittest.mock import MagicMock
        from orca_shared.registry.models import Model as ModelORM

        row = MagicMock(spec=ModelORM)
        row.model_id = model_id
        row.name = "resnet"
        row.architecture = "cnn"
        row.config = {}
        row.parameter_count = None
        row.flops = None

        execute_result = MagicMock()
        execute_result.scalar_one_or_none.return_value = row
        mock_session.execute.return_value = execute_result

        resp = await client.post("/api/v1/predict-performance", json=self._body(model_id))
        assert resp.status_code == 200

    async def test_response_fields(
        self,
        client: AsyncClient,
        model_id: UUID,
        mock_session: AsyncMock,
    ) -> None:
        from unittest.mock import MagicMock
        from orca_shared.registry.models import Model as ModelORM

        row = MagicMock(spec=ModelORM)
        row.model_id = model_id
        row.name = "mlp"
        row.architecture = "mlp"
        row.config = {}
        row.parameter_count = None
        row.flops = None

        execute_result = MagicMock()
        execute_result.scalar_one_or_none.return_value = row
        mock_session.execute.return_value = execute_result

        body = (
            await client.post("/api/v1/predict-performance", json=self._body(model_id))
        ).json()
        assert "predicted_score" in body
        assert "confidence" in body
        assert UUID(body["model_id"]) == model_id

    async def test_returns_404_when_model_not_found(
        self,
        client: AsyncClient,
        mock_session: AsyncMock,
    ) -> None:
        execute_result = MagicMock()
        execute_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = execute_result

        resp = await client.post(
            "/api/v1/predict-performance",
            json=self._body(uuid4()),
        )
        assert resp.status_code == 404

    async def test_returns_503_when_predictor_not_fitted(
        self,
        client: AsyncClient,
        model_id: UUID,
        mock_session: AsyncMock,
        mock_predictor: MagicMock,
    ) -> None:
        from unittest.mock import MagicMock
        from orca_shared.registry.models import Model as ModelORM

        row = MagicMock(spec=ModelORM)
        row.model_id = model_id
        row.name = "mlp"
        row.architecture = "mlp"
        row.config = {}
        row.parameter_count = None
        row.flops = None

        execute_result = MagicMock()
        execute_result.scalar_one_or_none.return_value = row
        mock_session.execute.return_value = execute_result

        mock_predictor.predict_with_confidence.side_effect = RuntimeError(
            "Predictor has not been fitted yet."
        )
        resp = await client.post("/api/v1/predict-performance", json=self._body(model_id))
        assert resp.status_code == 503

    async def test_returns_422_on_bad_body(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/predict-performance", json={})
        assert resp.status_code == 422


class TestSimilarTasks:
    def _body(self, top_k: int = 3) -> dict:
        return {"task_embedding": _EMBEDDING, "top_k": top_k}

    async def test_returns_200(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/similar-tasks", json=self._body())
        assert resp.status_code == 200

    async def test_returns_empty_list_when_index_empty(
        self, client: AsyncClient, mock_faiss_index: MagicMock
    ) -> None:
        mock_faiss_index.search.return_value = []
        body = (await client.post("/api/v1/similar-tasks", json=self._body())).json()
        assert body == []

    async def test_results_have_rank_starting_at_one(
        self, client: AsyncClient, mock_faiss_index: MagicMock
    ) -> None:
        tid1, tid2 = str(uuid4()), str(uuid4())
        mock_faiss_index.search.return_value = [(tid1, 0.95), (tid2, 0.80)]
        body = (await client.post("/api/v1/similar-tasks", json=self._body())).json()
        assert body[0]["rank"] == 1
        assert body[1]["rank"] == 2

    async def test_results_sorted_by_descending_score(
        self, client: AsyncClient, mock_faiss_index: MagicMock
    ) -> None:
        tid1, tid2 = str(uuid4()), str(uuid4())
        mock_faiss_index.search.return_value = [(tid1, 0.95), (tid2, 0.80)]
        body = (await client.post("/api/v1/similar-tasks", json=self._body())).json()
        assert body[0]["score"] > body[1]["score"]

    async def test_returns_503_when_faiss_not_loaded(
        self, client: AsyncClient
    ) -> None:
        from orcamind.api.deps import get_faiss_index
        from fastapi import HTTPException

        app = client._transport.app  # type: ignore[attr-defined]
        original = app.dependency_overrides.get(get_faiss_index)
        app.dependency_overrides[get_faiss_index] = lambda: (_ for _ in ()).throw(
            HTTPException(status_code=503, detail="FAISS index not loaded")
        )
        try:
            resp = await client.post("/api/v1/similar-tasks", json=self._body())
            assert resp.status_code == 503
        finally:
            if original is not None:
                app.dependency_overrides[get_faiss_index] = original
            else:
                del app.dependency_overrides[get_faiss_index]

    async def test_returns_422_when_top_k_below_minimum(
        self, client: AsyncClient
    ) -> None:
        resp = await client.post(
            "/api/v1/similar-tasks",
            json={"task_embedding": _EMBEDDING, "top_k": 0},
        )
        assert resp.status_code == 422
