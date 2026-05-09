"""Unit tests for FaissIndex."""

from __future__ import annotations

import numpy as np
import pytest

from orcamind.embedders.similarity import FaissIndex

_DIM = 64
_N = 100


def _random_l2_normalized(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


@pytest.fixture()
def populated_index() -> FaissIndex:
    index = FaissIndex(_DIM, metric="cosine")
    vecs = _random_l2_normalized(_N, _DIM, seed=7)
    for i, vec in enumerate(vecs):
        index.add(f"task_{i}", vec)
    return index


class TestSearch:
    def test_returns_k_results(self, populated_index: FaissIndex) -> None:
        query = _random_l2_normalized(1, _DIM, seed=99)[0]
        results = populated_index.search(query, k=10)
        assert len(results) == 10

    def test_scores_descending(self, populated_index: FaissIndex) -> None:
        query = _random_l2_normalized(1, _DIM, seed=55)[0]
        results = populated_index.search(query, k=15)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_k_larger_than_index(self) -> None:
        index = FaissIndex(_DIM, metric="cosine")
        vecs = _random_l2_normalized(5, _DIM, seed=1)
        for i, v in enumerate(vecs):
            index.add(f"t{i}", v)
        results = index.search(vecs[0], k=10)
        assert len(results) == 5

    def test_empty_index_returns_empty(self) -> None:
        index = FaissIndex(_DIM)
        query = _random_l2_normalized(1, _DIM)[0]
        assert index.search(query) == []

    def test_result_task_ids_are_strings(self, populated_index: FaissIndex) -> None:
        query = _random_l2_normalized(1, _DIM, seed=3)[0]
        results = populated_index.search(query, k=5)
        for task_id, _ in results:
            assert isinstance(task_id, str)


class TestSelfRankOne:
    def test_cosine_self_rank_one(self) -> None:
        index = FaissIndex(_DIM, metric="cosine")
        vecs = _random_l2_normalized(10, _DIM, seed=42)
        query = vecs[0].copy()
        index.add("task_q", query)
        for i, v in enumerate(vecs[1:], 1):
            index.add(f"task_{i}", v)

        results = index.search(query, k=10)
        assert results[0][0] == "task_q"
        assert abs(results[0][1] - 1.0) < 1e-5

    def test_l2_self_rank_one(self) -> None:
        index = FaissIndex(_DIM, metric="l2")
        vecs = _random_l2_normalized(10, _DIM, seed=42)
        query = vecs[0].copy()
        index.add("task_q", query)
        for i, v in enumerate(vecs[1:], 1):
            index.add(f"task_{i}", v)

        results = index.search(query, k=10)
        assert results[0][0] == "task_q"
        assert abs(results[0][1] - 0.0) < 1e-4


class TestSaveLoadRoundtrip:
    def test_roundtrip(self, tmp_path: object, populated_index: FaissIndex) -> None:
        from pathlib import Path

        save_path = str(Path(str(tmp_path)) / "idx")
        populated_index.save(save_path)

        fresh = FaissIndex(_DIM, metric="cosine")
        fresh.load(save_path)

        assert len(fresh) == _N

        query = _random_l2_normalized(1, _DIM, seed=200)[0]
        orig_results = populated_index.search(query, k=10)
        loaded_results = fresh.search(query, k=10)

        assert [tid for tid, _ in orig_results] == [tid for tid, _ in loaded_results]
        for (_, s1), (_, s2) in zip(orig_results, loaded_results):
            assert abs(s1 - s2) < 1e-5

    def test_roundtrip_l2(self, tmp_path: object) -> None:
        from pathlib import Path

        index = FaissIndex(_DIM, metric="l2")
        vecs = _random_l2_normalized(20, _DIM, seed=5)
        for i, v in enumerate(vecs):
            index.add(f"t{i}", v)

        save_path = str(Path(str(tmp_path)) / "l2_idx")
        index.save(save_path)

        fresh = FaissIndex(_DIM, metric="l2")
        fresh.load(save_path)

        assert len(fresh) == 20
        assert fresh._metric == "l2"


class TestLen:
    def test_empty_len(self) -> None:
        assert len(FaissIndex(_DIM)) == 0

    def test_incremental_len(self) -> None:
        index = FaissIndex(_DIM)
        vecs = _random_l2_normalized(3, _DIM, seed=0)
        for i, v in enumerate(vecs):
            index.add(f"t{i}", v)
            assert len(index) == i + 1

    def test_len_matches_n(self, populated_index: FaissIndex) -> None:
        assert len(populated_index) == _N


class TestMetrics:
    def test_cosine_index_type(self) -> None:
        import faiss

        assert isinstance(FaissIndex(_DIM, "cosine")._index, faiss.IndexFlatIP)

    def test_l2_index_type(self) -> None:
        import faiss

        assert isinstance(FaissIndex(_DIM, "l2")._index, faiss.IndexFlatL2)

    def test_invalid_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="metric"):
            FaissIndex(_DIM, "euclidean")

    def test_default_metric_is_cosine(self) -> None:
        import faiss

        assert isinstance(FaissIndex(_DIM)._index, faiss.IndexFlatIP)
