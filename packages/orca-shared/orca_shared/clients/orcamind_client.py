from __future__ import annotations

from uuid import UUID

import httpx

from orca_shared.clients._base import _BaseAsyncClient
from orca_shared.schemas.embedding import Embedding, SimilarityResult
from orca_shared.schemas.metrics import PerformanceMetrics
from orca_shared.schemas.model import ModelSummary
from orca_shared.schemas.recommendation import FeedbackRequest, ModelRecommendation, RecommendationRequest


class OrcaMindClient(_BaseAsyncClient):
    """Async httpx client for the OrcaMind meta-learning service."""

    async def recommend_model(self, req: RecommendationRequest) -> ModelRecommendation:
        response = await self._client.post(
            "/api/v1/recommend-model",
            json=req.model_dump(),
        )
        response.raise_for_status()
        items = response.json()
        if not isinstance(items, list) or not items:
            raise ValueError("OrcaMind returned no model recommendations")
        return ModelRecommendation.model_validate(items[0])

    async def predict_performance(
        self, task_embedding: list[float], model_id: UUID
    ) -> PerformanceMetrics:
        response = await self._client.post(
            "/api/v1/predict-performance",
            json={"task_embedding": task_embedding, "model_id": str(model_id)},
        )
        response.raise_for_status()
        data = response.json()
        return PerformanceMetrics(
            experiment_id=model_id,
            final_metrics={
                "predicted_score": data["predicted_score"],
                "confidence": data["confidence"],
            },
        )

    async def submit_feedback(self, req: FeedbackRequest) -> None:
        response = await self._client.post(
            "/api/v1/feedback",
            json=req.model_dump(mode="json"),
        )
        response.raise_for_status()

    async def embed_task(self, task_id: UUID) -> Embedding:
        response = await self._client.get(f"/api/v1/tasks/{task_id}/embedding")
        response.raise_for_status()
        return Embedding.model_validate(response.json())

    async def find_similar_tasks(
        self, embedding: list[float], top_k: int = 5
    ) -> list[SimilarityResult]:
        response = await self._client.post(
            "/api/v1/similar-tasks",
            json={"task_embedding": embedding, "top_k": top_k},
        )
        response.raise_for_status()
        return [SimilarityResult.model_validate(item) for item in response.json()]

    async def get_best_model(self, task_id: UUID) -> ModelSummary:
        emb = await self.embed_task(task_id)
        recommendation = await self.recommend_model(
            RecommendationRequest(task_embedding=emb.embedding_vector, top_k=1)
        )
        return ModelSummary(
            model_id=recommendation.model_id,
            name=recommendation.architecture or str(recommendation.model_id),
            architecture=recommendation.architecture,
        )
