"""Performance predictor model selector using a bootstrap XGBoost ensemble."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import xgboost as xgb

from orca_shared.schemas.model import ModelConfig
from orca_shared.schemas.recommendation import ModelRecommendation

from .base import ModelSelector
from .ranker import _build_feature_row


class PerformancePredictor(ModelSelector):
    """Recommends models by predicting performance with uncertainty estimation."""

    def __init__(self, n_estimators: int = 10, xgb_trees: int = 100) -> None:
        self._n_estimators = n_estimators
        self._xgb_trees = xgb_trees
        self._regressors: list[xgb.XGBRegressor] = []

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
        y = performances.astype(np.float64)

        rng = np.random.default_rng(0)
        self._regressors = []
        for _ in range(self._n_estimators):
            indices = rng.integers(0, n, size=n)
            reg = xgb.XGBRegressor(
                n_estimators=self._xgb_trees,
                tree_method="hist",
                verbosity=0,
            )
            reg.fit(X[indices], y[indices])
            self._regressors.append(reg)

    def _feature_row(self, task_embedding: np.ndarray, model: ModelConfig) -> np.ndarray:
        return _build_feature_row(task_embedding, model).reshape(1, -1)

    def predict_performance(self, task_embedding: np.ndarray, model_config: ModelConfig) -> float:
        """Return a point estimate of model performance in [0, 1]."""
        if not self._regressors:
            raise RuntimeError("Predictor has not been fitted yet.")
        X = self._feature_row(task_embedding, model_config)
        preds = np.array([float(reg.predict(X)[0]) for reg in self._regressors])
        return float(np.clip(preds.mean(), 0.0, 1.0))

    def predict_with_confidence(
        self, task_embedding: np.ndarray, model_config: ModelConfig
    ) -> tuple[float, float]:
        """Return (mean, std) of performance predictions across bootstrap ensemble."""
        if not self._regressors:
            raise RuntimeError("Predictor has not been fitted yet.")
        X = self._feature_row(task_embedding, model_config)
        preds = np.array([float(reg.predict(X)[0]) for reg in self._regressors])
        return float(np.clip(preds.mean(), 0.0, 1.0)), float(preds.std())

    def recommend(
        self,
        task_embedding: np.ndarray,
        candidate_models: list[ModelConfig],
        top_k: int = 3,
    ) -> list[ModelRecommendation]:
        if not self._regressors:
            raise RuntimeError("Predictor has not been fitted yet.")

        task_id = uuid4()
        scored: list[tuple[float, float, ModelConfig]] = []
        for model in candidate_models:
            mean, std = self.predict_with_confidence(task_embedding, model)
            scored.append((mean, std, model))

        scored.sort(key=lambda t: t[0], reverse=True)
        results: list[ModelRecommendation] = []
        for mean, std, model in scored[:top_k]:
            results.append(
                ModelRecommendation(
                    task_id=task_id,
                    model_id=model.model_id,
                    architecture=model.architecture,
                    predicted_score=mean,
                    confidence=float(1.0 / (1.0 + std)),
                    reasoning=f"bootstrap_std={std:.6f}",
                )
            )
        return results
