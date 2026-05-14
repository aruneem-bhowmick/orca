"""Full pipeline integration test: StatisticalEmbedder + FaissIndex + NearestNeighborSelector.

No mocking — all components run against real implementations.
No external services required (no Docker, no database, no network calls).
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from orca_shared.schemas.model import ModelConfig
from orca_shared.schemas.recommendation import ModelRecommendation
from orcamind.embedders.similarity import FaissIndex
from orcamind.embedders.statistical import StatisticalEmbedder
from orcamind.selectors.nearest_neighbor import NearestNeighborSelector

_N_TASKS = 10
_N_MODELS = 5
_BEST_MODEL_IDX = 0
_BEST_PERF = 0.9
_OTHER_PERF = 0.1
_EMBED_DIM = 25  # StatisticalEmbedder always outputs 25-dim
_SEED = 77


def _make_synthetic_dataframe(seed: int, n_rows: int = 50) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "f0": rng.normal(seed * 0.1, 1.0, n_rows),
            "f1": rng.normal(0.0, seed * 0.05 + 0.5, n_rows),
            "f2": rng.uniform(0, 10, n_rows),
            "f3": rng.exponential(1.0, n_rows),
            "f4": rng.standard_normal(n_rows),
        }
    )
    y = pd.Series(rng.integers(0, 3, n_rows), name="target")
    return X, y


def _make_model_configs() -> list[ModelConfig]:
    return [
        ModelConfig(
            model_id=uuid4(),
            name=f"pipeline_model_{i}",
            architecture=f"arch_{i}",
            config={
                "hidden_size": 64 if i == _BEST_MODEL_IDX else 256,
                "num_layers": 2,
                "lr": 1e-3,
                "batch_size": 32,
            },
            parameter_count=100_000 * (i + 1),
            flops=1_000_000 * (i + 1),
        )
        for i in range(_N_MODELS)
    ]


@pytest.fixture(scope="module")
def pipeline_components() -> (
    tuple[StatisticalEmbedder, FaissIndex, NearestNeighborSelector, list[ModelConfig], np.ndarray]
):
    """Build and return all fitted pipeline components.

    Returns (embedder, faiss_index, selector, model_configs, query_embedding).
    Module-scoped so the XGBoost/FAISS fitting runs once per test session.
    """
    embedder = StatisticalEmbedder()
    model_configs = _make_model_configs()

    task_embeddings: list[np.ndarray] = []
    task_ids: list[str] = []
    for i in range(_N_TASKS):
        df, y = _make_synthetic_dataframe(seed=_SEED + i)
        emb = embedder.embed(df, y)  # shape (25,)
        task_embeddings.append(emb)
        task_ids.append(str(uuid4()))

    faiss_index = FaissIndex(_EMBED_DIM, metric="cosine")
    for tid, emb in zip(task_ids, task_embeddings):
        faiss_index.add(tid, emb)

    flat_embs: list[np.ndarray] = []
    flat_configs: list[ModelConfig] = []
    flat_perfs: list[float] = []
    for i in range(_N_TASKS):
        for j, cfg in enumerate(model_configs):
            flat_embs.append(task_embeddings[i].copy())
            flat_configs.append(cfg)
            flat_perfs.append(_BEST_PERF if j == _BEST_MODEL_IDX else _OTHER_PERF)

    selector = NearestNeighborSelector()
    selector.fit(
        np.array(flat_embs),
        flat_configs,
        np.array(flat_perfs, dtype=np.float64),
    )

    query_df, query_y = _make_synthetic_dataframe(seed=_SEED + _N_TASKS + 1)
    query_emb = embedder.embed(query_df, query_y)

    return embedder, faiss_index, selector, model_configs, query_emb


class TestFullPipelineComponents:
    """Sanity-check each component after construction."""

    def test_embedder_produces_25_dim_vector(
        self, pipeline_components: tuple
    ) -> None:
        embedder, *_ = pipeline_components
        df, y = _make_synthetic_dataframe(seed=0)
        emb = embedder.embed(df, y)
        assert emb.shape == (_EMBED_DIM,)

    def test_faiss_index_contains_n_tasks(
        self, pipeline_components: tuple
    ) -> None:
        _, faiss_index, *_ = pipeline_components
        assert len(faiss_index) == _N_TASKS

    def test_faiss_index_search_returns_results(
        self, pipeline_components: tuple
    ) -> None:
        _, faiss_index, _, _, query_emb = pipeline_components
        results = faiss_index.search(query_emb, k=3)
        assert len(results) == 3
        for task_id, score in results:
            assert isinstance(task_id, str)
            assert isinstance(score, float)

    def test_selector_fitted_without_error(
        self, pipeline_components: tuple
    ) -> None:
        _, _, selector, model_configs, query_emb = pipeline_components
        recs = selector.recommend(query_emb, model_configs, top_k=3)
        assert len(recs) == 3


class TestFullPipelineRecommendation:
    """End-to-end: recommend models for a query task and validate the result."""

    def test_best_model_in_top3_recommendations(
        self, pipeline_components: tuple
    ) -> None:
        _, _, selector, model_configs, query_emb = pipeline_components
        best_model_id = model_configs[_BEST_MODEL_IDX].model_id
        recs = selector.recommend(query_emb, model_configs, top_k=3)
        returned_ids = [r.model_id for r in recs]
        assert best_model_id in returned_ids, (
            f"Best model {best_model_id} not found in top-3: {returned_ids}"
        )

    def test_recommendations_are_model_recommendation_objects(
        self, pipeline_components: tuple
    ) -> None:
        _, _, selector, model_configs, query_emb = pipeline_components
        recs = selector.recommend(query_emb, model_configs, top_k=3)
        assert all(isinstance(r, ModelRecommendation) for r in recs)

    def test_recommendations_sorted_by_descending_score(
        self, pipeline_components: tuple
    ) -> None:
        _, _, selector, model_configs, query_emb = pipeline_components
        recs = selector.recommend(query_emb, model_configs, top_k=3)
        scores = [r.predicted_score for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_training_embeddings_are_all_finite(
        self, pipeline_components: tuple
    ) -> None:
        embedder, *_ = pipeline_components
        for i in range(_N_TASKS):
            df, y = _make_synthetic_dataframe(seed=_SEED + i)
            emb = embedder.embed(df, y)
            assert np.all(np.isfinite(emb)), f"Non-finite embedding for task seed={_SEED + i}"
