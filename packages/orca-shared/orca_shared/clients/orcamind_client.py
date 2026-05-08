from __future__ import annotations

from uuid import UUID

from orca_shared.clients._base import _BaseAsyncClient
from orca_shared.schemas.embedding import Embedding, SimilarityResult
from orca_shared.schemas.metrics import PerformanceMetrics
from orca_shared.schemas.model import ModelSummary
from orca_shared.schemas.recommendation import FeedbackRequest, ModelRecommendation, RecommendationRequest


class OrcaMindClient(_BaseAsyncClient):
    """Async httpx client for the OrcaMind meta-learning service."""

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
