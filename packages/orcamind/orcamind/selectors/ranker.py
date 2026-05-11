"""Learning-to-rank model selector using XGBoost pairwise ranking."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import xgboost as xgb

from orca_shared.schemas.model import ModelConfig
from orca_shared.schemas.recommendation import ModelRecommendation

from .base import ModelSelector

_CONFIG_KEYS = ("num_layers", "hidden_size", "lr", "batch_size")


def _model_features(model: ModelConfig) -> np.ndarray:
    cfg = model.config or {}
    numeric = [
        float(model.parameter_count or 0),
        float(model.flops or 0),
        *(float(cfg.get(k, 0) or 0) for k in _CONFIG_KEYS),
    ]
    return np.array(numeric, dtype=np.float64)


def _build_feature_row(task_embedding: np.ndarray, model: ModelConfig) -> np.ndarray:
    return np.concatenate([task_embedding.astype(np.float64), _model_features(model)])


class LearningToRankSelector(ModelSelector):
    """Recommends models using XGBoost pairwise ranking."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1) -> None:
        self._ranker = xgb.XGBRanker(
            objective="rank:pairwise",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            tree_method="hist",
            verbosity=0,
        )
        self._fitted = False

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
        n = len(task_embeddings)
        X = np.stack(
            [_build_feature_row(task_embeddings[i], model_configs[i]) for i in range(n)]
        )
        # Assign all samples to a single query group (qid=0) for pairwise ranking.
        y = performances.astype(np.float64)
        self._ranker.fit(X, y, qid=np.zeros(n, dtype=np.int32))
        self._fitted = True

    def recommend(
        self,
        task_embedding: np.ndarray,
        candidate_models: list[ModelConfig],
        top_k: int = 3,
    ) -> list[ModelRecommendation]:
        if not self._fitted:
            raise RuntimeError("Selector has not been fitted yet.")

        task_id = uuid4()
        X_cand = np.stack(
            [_build_feature_row(task_embedding, m) for m in candidate_models]
        )
        scores = self._ranker.predict(X_cand)

        order = np.argsort(scores)[::-1]
        results: list[ModelRecommendation] = []
        for idx in order[:top_k]:
            model = candidate_models[int(idx)]
            results.append(
                ModelRecommendation(
                    task_id=task_id,
                    model_id=model.model_id,
                    architecture=model.architecture,
                    predicted_score=float(scores[int(idx)]),
                    confidence=None,
                )
            )
        return results
