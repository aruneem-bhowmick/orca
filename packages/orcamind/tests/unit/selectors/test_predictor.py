"""Unit tests for PerformancePredictor."""

from __future__ import annotations

import numpy as np
import pytest

from orca_shared.schemas.model import ModelConfig
from orca_shared.schemas.recommendation import ModelRecommendation
from orcamind.selectors.predictor import PerformancePredictor

BEST_MODEL_IDX = 0
EMBED_DIM = 32
N_MODELS = 5


@pytest.fixture()
def fitted_predictor(
    training_data: tuple[np.ndarray, list[ModelConfig], np.ndarray],
) -> PerformancePredictor:
    pred = PerformancePredictor(n_estimators=10, xgb_trees=50)
    embs, configs, perfs = training_data
    pred.fit(embs, configs, perfs)
    return pred


class TestTopKCount:
    @pytest.mark.parametrize("top_k", [1, 2, 3, N_MODELS])
    def test_returns_exactly_top_k(
        self,
        fitted_predictor: PerformancePredictor,
        model_configs: list[ModelConfig],
        top_k: int,
    ) -> None:
        query = np.random.default_rng(30).standard_normal(EMBED_DIM)
        recs = fitted_predictor.recommend(query, model_configs, top_k=top_k)
        assert len(recs) == top_k

    def test_recommendation_type(
        self,
        fitted_predictor: PerformancePredictor,
        model_configs: list[ModelConfig],
    ) -> None:
        query = np.random.default_rng(31).standard_normal(EMBED_DIM)
        recs = fitted_predictor.recommend(query, model_configs, top_k=3)
        assert all(isinstance(r, ModelRecommendation) for r in recs)


class TestBestModelSelection:
    def test_best_model_ranked_first_for_held_out_tasks(
        self,
        fitted_predictor: PerformancePredictor,
        model_configs: list[ModelConfig],
        held_out_embeddings: np.ndarray,
    ) -> None:
        best_id = model_configs[BEST_MODEL_IDX].model_id
        hits = 0
        for emb in held_out_embeddings:
            recs = fitted_predictor.recommend(emb, model_configs, top_k=1)
            if recs[0].model_id == best_id:
                hits += 1
        # Must beat random baseline (1/N_MODELS = 0.2)
        assert hits / len(held_out_embeddings) > 1.0 / N_MODELS


class TestPredictPerformance:
    def test_output_in_unit_interval(
        self,
        fitted_predictor: PerformancePredictor,
        model_configs: list[ModelConfig],
    ) -> None:
        rng = np.random.default_rng(40)
        for _ in range(10):
            emb = rng.standard_normal(EMBED_DIM)
            for model in model_configs:
                score = fitted_predictor.predict_performance(emb, model)
                assert 0.0 <= score <= 1.0, f"score={score} outside [0, 1]"

    def test_output_is_float(
        self,
        fitted_predictor: PerformancePredictor,
        model_configs: list[ModelConfig],
    ) -> None:
        emb = np.random.default_rng(41).standard_normal(EMBED_DIM)
        score = fitted_predictor.predict_performance(emb, model_configs[0])
        assert isinstance(score, float)

    def test_raises_before_fit(self, model_configs: list[ModelConfig]) -> None:
        pred = PerformancePredictor()
        with pytest.raises(RuntimeError, match="fitted"):
            pred.predict_performance(np.zeros(EMBED_DIM), model_configs[0])


class TestPredictWithConfidence:
    def test_returns_two_floats(
        self,
        fitted_predictor: PerformancePredictor,
        model_configs: list[ModelConfig],
    ) -> None:
        emb = np.random.default_rng(50).standard_normal(EMBED_DIM)
        result = fitted_predictor.predict_with_confidence(emb, model_configs[0])
        assert isinstance(result, tuple) and len(result) == 2
        mean, std = result
        assert isinstance(mean, float) and isinstance(std, float)

    def test_mean_in_unit_interval(
        self,
        fitted_predictor: PerformancePredictor,
        model_configs: list[ModelConfig],
    ) -> None:
        emb = np.random.default_rng(51).standard_normal(EMBED_DIM)
        mean, _ = fitted_predictor.predict_with_confidence(emb, model_configs[0])
        assert 0.0 <= mean <= 1.0

    def test_std_is_non_negative(
        self,
        fitted_predictor: PerformancePredictor,
        model_configs: list[ModelConfig],
    ) -> None:
        emb = np.random.default_rng(52).standard_normal(EMBED_DIM)
        _, std = fitted_predictor.predict_with_confidence(emb, model_configs[0])
        assert std >= 0.0

    def test_raises_before_fit(self, model_configs: list[ModelConfig]) -> None:
        pred = PerformancePredictor()
        with pytest.raises(RuntimeError, match="fitted"):
            pred.predict_with_confidence(np.zeros(EMBED_DIM), model_configs[0])


class TestConfidenceInRecommendations:
    def test_confidence_in_unit_interval(
        self,
        fitted_predictor: PerformancePredictor,
        model_configs: list[ModelConfig],
    ) -> None:
        emb = np.random.default_rng(60).standard_normal(EMBED_DIM)
        recs = fitted_predictor.recommend(emb, model_configs, top_k=3)
        for r in recs:
            assert r.confidence is not None and 0.0 < r.confidence <= 1.0

    def test_reasoning_contains_std(
        self,
        fitted_predictor: PerformancePredictor,
        model_configs: list[ModelConfig],
    ) -> None:
        emb = np.random.default_rng(61).standard_normal(EMBED_DIM)
        recs = fitted_predictor.recommend(emb, model_configs, top_k=1)
        assert recs[0].reasoning is not None and "bootstrap_std=" in recs[0].reasoning


class TestFitValidation:
    def test_raises_on_length_mismatch(self, model_configs: list[ModelConfig]) -> None:
        pred = PerformancePredictor()
        embs = np.zeros((5, EMBED_DIM))
        perfs = np.zeros(4)
        with pytest.raises(ValueError, match="same length"):
            pred.fit(embs, model_configs[:5], perfs)

    def test_raises_before_fit(self, model_configs: list[ModelConfig]) -> None:
        pred = PerformancePredictor()
        with pytest.raises(RuntimeError, match="fitted"):
            pred.recommend(np.zeros(EMBED_DIM), model_configs, top_k=1)
