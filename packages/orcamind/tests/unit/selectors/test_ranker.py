"""Unit tests for LearningToRankSelector."""

from __future__ import annotations

import numpy as np
import pytest

from orca_shared.schemas.model import ModelConfig
from orca_shared.schemas.recommendation import ModelRecommendation
from orcamind.selectors.ranker import LearningToRankSelector

BEST_MODEL_IDX = 0
EMBED_DIM = 32
N_MODELS = 5


@pytest.fixture()
def fitted_selector(
    training_data: tuple[np.ndarray, list[ModelConfig], np.ndarray],
) -> LearningToRankSelector:
    sel = LearningToRankSelector(n_estimators=50)
    embs, configs, perfs = training_data
    sel.fit(embs, configs, perfs)
    return sel


class TestTopKCount:
    @pytest.mark.parametrize("top_k", [1, 2, 3, N_MODELS])
    def test_returns_exactly_top_k(
        self,
        fitted_selector: LearningToRankSelector,
        model_configs: list[ModelConfig],
        top_k: int,
    ) -> None:
        query = np.random.default_rng(10).standard_normal(EMBED_DIM)
        recs = fitted_selector.recommend(query, model_configs, top_k=top_k)
        assert len(recs) == top_k

    def test_recommendation_type(
        self,
        fitted_selector: LearningToRankSelector,
        model_configs: list[ModelConfig],
    ) -> None:
        query = np.random.default_rng(11).standard_normal(EMBED_DIM)
        recs = fitted_selector.recommend(query, model_configs, top_k=3)
        assert all(isinstance(r, ModelRecommendation) for r in recs)


class TestBestModelSelection:
    def test_best_model_ranked_first_for_held_out_tasks(
        self,
        fitted_selector: LearningToRankSelector,
        model_configs: list[ModelConfig],
        held_out_embeddings: np.ndarray,
    ) -> None:
        best_id = model_configs[BEST_MODEL_IDX].model_id
        hits = 0
        for emb in held_out_embeddings:
            recs = fitted_selector.recommend(emb, model_configs, top_k=1)
            if recs[0].model_id == best_id:
                hits += 1
        # Must beat random baseline (1/N_MODELS = 0.2)
        assert hits / len(held_out_embeddings) > 1.0 / N_MODELS


class TestScores:
    def test_predicted_scores_are_finite(
        self,
        fitted_selector: LearningToRankSelector,
        model_configs: list[ModelConfig],
    ) -> None:
        query = np.random.default_rng(20).standard_normal(EMBED_DIM)
        recs = fitted_selector.recommend(query, model_configs, top_k=3)
        assert all(np.isfinite(r.predicted_score) for r in recs)

    def test_unique_model_ids_in_results(
        self,
        fitted_selector: LearningToRankSelector,
        model_configs: list[ModelConfig],
    ) -> None:
        query = np.random.default_rng(21).standard_normal(EMBED_DIM)
        recs = fitted_selector.recommend(query, model_configs, top_k=3)
        model_ids = [r.model_id for r in recs]
        assert len(model_ids) == len(set(model_ids))


class TestFitValidation:
    def test_raises_on_length_mismatch(self, model_configs: list[ModelConfig]) -> None:
        sel = LearningToRankSelector()
        embs = np.zeros((5, EMBED_DIM))
        perfs = np.zeros(4)
        with pytest.raises(ValueError, match="same length"):
            sel.fit(embs, model_configs[:5], perfs)

    def test_raises_before_fit(self, model_configs: list[ModelConfig]) -> None:
        sel = LearningToRankSelector()
        with pytest.raises(RuntimeError, match="fitted"):
            sel.recommend(np.zeros(EMBED_DIM), model_configs, top_k=1)
