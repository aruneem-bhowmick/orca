"""Unit tests for PerformancePredictor."""

from __future__ import annotations

import re
from uuid import uuid4

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

    def test_recommendations_are_sorted_by_predicted_mean(
        self,
        fitted_predictor: PerformancePredictor,
        model_configs: list[ModelConfig],
    ) -> None:
        query = np.random.default_rng(32).standard_normal(EMBED_DIM)
        recs = fitted_predictor.recommend(query, model_configs, top_k=3)
        means = {
            cfg.model_id: fitted_predictor.predict_with_confidence(query, cfg)[0]
            for cfg in model_configs
        }
        returned = [means[r.model_id] for r in recs]
        assert returned == sorted(returned, reverse=True)


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
        assert recs[0].reasoning is not None
        match = re.search(
            r"bootstrap_std=([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", recs[0].reasoning
        )
        assert match is not None
        assert float(match.group(1)) >= 0.0


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


class TestSpearmanRankCorrelation:
    """200 synthetic tasks → fit on 160 → evaluate on 40 → mean Spearman > 0.7."""

    _N_TRAIN_TASKS = 160
    _N_TEST_TASKS = 40
    _N_MODELS = 5
    _EMBED_DIM = EMBED_DIM  # 32, module-level constant
    _SEED = 99              # distinct from conftest SEED=42
    _BASE_PERFS = [0.80, 0.60, 0.45, 0.30, 0.15]
    _NOISE_SCALE = 0.05
    _THRESHOLD = 0.7

    @pytest.fixture(scope="class")
    def _configs(self) -> list[ModelConfig]:
        return [
            ModelConfig(
                model_id=uuid4(),
                name=f"s_model_{i}",
                architecture=f"s_arch_{i}",
                config={"hidden_size": 64, "num_layers": 2, "lr": 1e-3, "batch_size": 32},
                parameter_count=100_000 * (i + 1),
                flops=1_000_000 * (i + 1),
            )
            for i in range(self._N_MODELS)
        ]

    @pytest.fixture(scope="class")
    def _fitted(
        self, _configs: list[ModelConfig]
    ) -> tuple[PerformancePredictor, np.ndarray]:
        rng = np.random.default_rng(self._SEED)
        all_embs = rng.standard_normal((200, self._EMBED_DIM))

        train_embs: list[np.ndarray] = []
        train_cfgs: list[ModelConfig] = []
        train_perfs: list[float] = []
        for emb in all_embs[: self._N_TRAIN_TASKS]:
            for idx, cfg in enumerate(_configs):
                noise = float(rng.normal(0, self._NOISE_SCALE))
                train_embs.append(emb.copy())
                train_cfgs.append(cfg)
                train_perfs.append(float(np.clip(self._BASE_PERFS[idx] + noise, 0.0, 1.0)))

        pred = PerformancePredictor(n_estimators=10, xgb_trees=50)
        pred.fit(np.array(train_embs), train_cfgs, np.array(train_perfs))

        test_embs = all_embs[self._N_TRAIN_TASKS :]  # shape (40, 32)
        return pred, test_embs

    def test_mean_spearman_exceeds_threshold(
        self,
        _configs: list[ModelConfig],
        _fitted: tuple[PerformancePredictor, np.ndarray],
    ) -> None:
        from scipy.stats import spearmanr

        pred, test_embs = _fitted
        gt = self._BASE_PERFS  # deterministic ground-truth ranking per model

        correlations: list[float] = []
        for emb in test_embs:
            predicted = [pred.predict_performance(emb, cfg) for cfg in _configs]
            corr = spearmanr(predicted, gt).statistic
            correlations.append(float(corr))

        mean_corr = float(np.mean(correlations))
        assert mean_corr > self._THRESHOLD, (
            f"Mean Spearman {mean_corr:.4f} did not exceed threshold {self._THRESHOLD}"
        )
