"""Nearest-neighbor model selector based on cosine similarity."""

from __future__ import annotations

from uuid import uuid4

import numpy as np

from orca_shared.schemas.model import ModelConfig
from orca_shared.schemas.recommendation import ModelRecommendation

from .base import ModelSelector


class NearestNeighborSelector(ModelSelector):
    """Recommends models by aggregating performance of similar historical tasks."""

    def __init__(self) -> None:
        self._embeddings: list[np.ndarray] = []
        self._configs: list[ModelConfig] = []
        self._performances: list[float] = []

    def fit(
        self,
        task_embeddings: np.ndarray,
        model_configs: list[ModelConfig],
        performances: np.ndarray,
    ) -> None:
        if len(task_embeddings) != len(model_configs) or len(task_embeddings) != len(performances):
            raise ValueError(
                "task_embeddings, model_configs, and performances must all have the same length"
            )
        self._embeddings = [task_embeddings[i] for i in range(len(task_embeddings))]
        self._configs = list(model_configs)
        self._performances = list(performances.astype(float))

    def recommend(
        self,
        task_embedding: np.ndarray,
        candidate_models: list[ModelConfig],
        top_k: int = 3,
    ) -> list[ModelRecommendation]:
        if not self._embeddings:
            raise RuntimeError("Selector has not been fitted yet.")

        task_id = uuid4()
        k_anchor = min(len(self._embeddings), max(top_k, 5))

        stored = np.array(self._embeddings)
        norms_stored = np.linalg.norm(stored, axis=1, keepdims=True)
        norms_stored = np.where(norms_stored == 0, 1.0, norms_stored)
        stored_normed = stored / norms_stored

        query_norm = np.linalg.norm(task_embedding)
        query_normed = task_embedding / (query_norm if query_norm > 0 else 1.0)

        similarities = stored_normed @ query_normed
        anchor_indices = np.argsort(similarities)[-k_anchor:][::-1]
        anchor_sims = similarities[anchor_indices]

        scored: list[tuple[float, float, ModelConfig]] = []
        for candidate in candidate_models:
            weighted_sum = 0.0
            sim_sum = 0.0
            for idx, sim in zip(anchor_indices, anchor_sims, strict=True):
                if self._configs[idx].model_id == candidate.model_id:
                    weighted_sum += float(sim) * self._performances[idx]
                    sim_sum += float(sim)
            if sim_sum > 0:
                score = weighted_sum / sim_sum
                confidence = float(sim_sum / max(anchor_sims.sum(), 1e-9))
            else:
                score = 0.0
                confidence = 0.0
            scored.append((score, confidence, candidate))

        scored.sort(key=lambda t: t[0], reverse=True)
        results: list[ModelRecommendation] = []
        for score, confidence, model in scored[:top_k]:
            results.append(
                ModelRecommendation(
                    task_id=task_id,
                    model_id=model.model_id,
                    architecture=model.architecture,
                    predicted_score=float(score),
                    confidence=float(confidence),
                )
            )
        return results
