"""Abstract base class for model selectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from orca_shared.schemas.model import ModelConfig
from orca_shared.schemas.recommendation import ModelRecommendation


class ModelSelector(ABC):
    """Abstract base for all model selection strategies."""

    @abstractmethod
    def recommend(
        self,
        task_embedding: np.ndarray,
        candidate_models: list[ModelConfig],
        top_k: int = 3,
    ) -> list[ModelRecommendation]: ...

    @abstractmethod
    def fit(
        self,
        task_embeddings: np.ndarray,
        model_configs: list[ModelConfig],
        performances: np.ndarray,
    ) -> None: ...
