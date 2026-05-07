from __future__ import annotations

from uuid import UUID

import httpx

from orca_shared.schemas.embedding import Embedding, SimilarityResult
from orca_shared.schemas.metrics import PerformanceMetrics
from orca_shared.schemas.model import ModelSummary
from orca_shared.schemas.recommendation import FeedbackRequest, ModelRecommendation, RecommendationRequest


class OrcaMindClient:
    """Async httpx client for the OrcaMind meta-learning service."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=20),
        )

    async def recommend_model(self, req: RecommendationRequest) -> ModelRecommendation:
        raise NotImplementedError("OrcaMindClient.recommend_model is not yet implemented")

    async def predict_performance(
        self, task_embedding: list[float], model_id: UUID
    ) -> PerformanceMetrics:
        raise NotImplementedError("OrcaMindClient.predict_performance is not yet implemented")

    async def submit_feedback(self, req: FeedbackRequest) -> None:
        raise NotImplementedError("OrcaMindClient.submit_feedback is not yet implemented")

    async def get_best_model(self, task_id: UUID) -> ModelSummary:
        raise NotImplementedError("OrcaMindClient.get_best_model is not yet implemented")

    async def embed_task(self, task_id: UUID) -> Embedding:
        raise NotImplementedError("OrcaMindClient.embed_task is not yet implemented")

    async def find_similar_tasks(
        self, embedding: list[float], top_k: int = 5
    ) -> list[SimilarityResult]:
        raise NotImplementedError("OrcaMindClient.find_similar_tasks is not yet implemented")

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OrcaMindClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
