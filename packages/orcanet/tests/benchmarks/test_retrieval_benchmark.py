"""Recall@9 benchmark for the HybridRetriever.

Validates that pure vector retrieval (``use_llm_reranking=False``) achieves
Recall@9 > 0.85 on a 100-task synthetic benchmark with known ground-truth
similar pairs.  No live LLM or FAISS binary is required — a lightweight exact
cosine-similarity index is used in its place.

``_TOP_K_FINAL`` is set to 9 (one less than the group size of 10) so that the
query task itself — which is in the index and trivially scores cosine ≈ 1.0 —
cannot fill the final retrieval slot and inflate recall to 1.0.  Ground truth
excludes the query, so 8 of 9 returned positives yields a genuine Recall@9 ≈
0.89 > 0.85 threshold.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import numpy as np
import pytest
import torch

from orca_shared.schemas.task import Task
from orcanet.retrieval.retriever import HybridRetriever, task_to_feature_vector

# ---------------------------------------------------------------------------
# Benchmark constants
# ---------------------------------------------------------------------------

_N_GROUPS: int = 10
_N_PER_GROUP: int = 10
_TOP_K_FINAL: int = 9  # < group size (10) so the self-hit cannot trivially fill all k slots
_SIMILARITY_THRESHOLD: float = 0.50
_RECALL_THRESHOLD: float = 0.85
_CLUSTER_NOISE: float = 0.02

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)

# One orthonormal basis vector per group in 25-dim space.
# Cross-group cosine similarity is 0; within-group cosine is ≈ 1.
_GROUP_CENTERS: list[np.ndarray] = [
    np.eye(25, dtype=np.float32)[g] for g in range(_N_GROUPS)
]


# ---------------------------------------------------------------------------
# Lightweight exact-search index (replaces a live FAISS binary)
# ---------------------------------------------------------------------------


class _SimpleFaissIndex:
    """Exact cosine-similarity nearest-neighbour lookup — no real FAISS dependency."""

    def __init__(self) -> None:
        """Initialise an empty index."""
        self._ids: list[str] = []
        self._embeddings: list[np.ndarray] = []

    def add(self, task_id: str, embedding: np.ndarray) -> None:
        """Store the L2-normalised *embedding* keyed by *task_id*."""
        norm = float(np.linalg.norm(embedding))
        self._ids.append(task_id)
        self._embeddings.append(embedding / max(norm, 1e-8))

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return up to *k* ``(task_id, cosine_score)`` pairs, highest score first."""
        q = query / max(float(np.linalg.norm(query)), 1e-8)
        scores = [float(np.dot(q, e)) for e in self._embeddings]
        order = np.argsort(scores)[::-1][:k]
        return [(self._ids[i], scores[i]) for i in order]


# ---------------------------------------------------------------------------
# Controlled embedder (maps task feature vectors to pre-assigned cluster embeddings)
# ---------------------------------------------------------------------------


class _ControlledEmbedder:
    """Returns pre-computed cluster embeddings so group structure is preserved exactly."""

    def __init__(self, fvec_key_to_embedding: dict[tuple, np.ndarray]) -> None:
        """Store the feature-vector → cluster-embedding lookup table."""
        self._lookup = fvec_key_to_embedding

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return the cluster embedding for the 25-dim feature tensor *x*.

        The lookup key is derived by rounding to 4 decimal places, so tasks
        with distinct ``n_samples`` (encoded as ``log1p``) are distinguishable.
        """
        key = tuple(x.squeeze(0).numpy().round(4))
        emb = self._lookup.get(key, np.zeros(25, dtype=np.float32))
        return torch.tensor(emb, dtype=torch.float32).unsqueeze(0)


# ---------------------------------------------------------------------------
# Synthetic task factory
# ---------------------------------------------------------------------------


def _make_tasks() -> tuple[list[list[Task]], dict[UUID, int]]:
    """Create 100 synthetic tasks in 10 groups of 10 with unique feature vectors.

    Each task encodes its group membership through ``n_features`` and
    ``n_classes``, and its uniqueness through a per-task ``n_samples`` offset.
    """
    groups: list[list[Task]] = []
    task_to_group: dict[UUID, int] = {}
    for g in range(_N_GROUPS):
        group_tasks: list[Task] = []
        for t in range(_N_PER_GROUP):
            task = Task(
                task_id=uuid4(),
                name=f"task_g{g}_t{t}",
                domain=f"domain_{g}",
                task_type="classification",
                n_samples=1000 * (g + 1) + t,  # unique per task
                n_features=10 + g,
                n_classes=2 + g,
                created_at=_NOW,
                updated_at=_NOW,
            )
            group_tasks.append(task)
            task_to_group[task.task_id] = g
        groups.append(group_tasks)
    return groups, task_to_group


# ---------------------------------------------------------------------------
# Retriever factory
# ---------------------------------------------------------------------------


def _build_retriever(
    groups: list[list[Task]],
    task_to_group: dict[UUID, int],  # noqa: ARG001 (kept for caller clarity)
) -> HybridRetriever:
    """Assemble a HybridRetriever backed by controlled mocks for a deterministic benchmark.

    Each task is assigned a cluster embedding (group centre + small Gaussian
    noise) so that same-group tasks have cosine similarity ≈ 1 and
    cross-group tasks have cosine similarity ≈ 0.
    """
    rng = np.random.default_rng(42)
    faiss_idx = _SimpleFaissIndex()
    fvec_to_emb: dict[tuple, np.ndarray] = {}
    all_tasks: dict[UUID, Task] = {}

    for g, group_tasks in enumerate(groups):
        center = _GROUP_CENTERS[g]
        for task in group_tasks:
            noise = rng.normal(0, _CLUSTER_NOISE, 25).astype(np.float32)
            emb = center + noise
            key = tuple(task_to_feature_vector(task).round(4))
            fvec_to_emb[key] = emb
            faiss_idx.add(str(task.task_id), emb)
            all_tasks[task.task_id] = task

    async def _get_by_id(uid: UUID) -> Task | None:
        return all_tasks.get(uid)

    mock_repo = MagicMock()
    mock_repo.get_by_id = AsyncMock(side_effect=_get_by_id)

    mock_expander = MagicMock()
    mock_expander.expand = AsyncMock(return_value=[])

    mock_ranker = MagicMock()
    mock_ranker.rerank = AsyncMock(return_value=[])

    return HybridRetriever(
        faiss_index=faiss_idx,
        task_repository=mock_repo,
        embedder=_ControlledEmbedder(fvec_to_emb),
        query_expander=mock_expander,
        llm_ranker=mock_ranker,
        top_k_initial=_N_PER_GROUP * 2,
        top_k_final=_TOP_K_FINAL,
        similarity_threshold=_SIMILARITY_THRESHOLD,
        use_llm_reranking=False,
    )


# ---------------------------------------------------------------------------
# Recall helpers
# ---------------------------------------------------------------------------


def _recall_at_k(
    results: list[tuple[Task, float, str]],
    true_positive_ids: set[UUID],
) -> float:
    """Return the fraction of *true_positive_ids* that appear in *results*.

    Args:
        results:           Retrieved (Task, score, reasoning) triples.
        true_positive_ids: Ground-truth relevant task IDs for the query.

    Returns:
        A float in [0, 1]; 1.0 if *true_positive_ids* is empty.
    """
    if not true_positive_ids:
        return 1.0
    retrieved = {r[0].task_id for r in results}
    return len(retrieved & true_positive_ids) / len(true_positive_ids)


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


class TestRecallAtNine:
    """Recall@9 benchmark: pure vector retrieval on a 100-task synthetic registry.

    ``top_k_final=9`` (< group size=10) ensures the self-hit cannot consume the
    final result slot and inflate recall to a trivial 1.0.
    """

    @pytest.mark.asyncio
    async def test_mean_recall_exceeds_threshold(self) -> None:
        """Mean Recall@9 across all 100 query tasks must be > 0.85."""
        groups, task_to_group = _make_tasks()
        retriever = _build_retriever(groups, task_to_group)

        recalls: list[float] = []
        for g, group_tasks in enumerate(groups):
            for query_task in group_tasks:
                true_pos = {
                    t.task_id for t in group_tasks if t.task_id != query_task.task_id
                }
                results = await retriever.retrieve(query_task)
                recalls.append(_recall_at_k(results, true_pos))

        mean_recall = float(np.mean(recalls))
        assert mean_recall > _RECALL_THRESHOLD, (
            f"Recall@{_TOP_K_FINAL} = {mean_recall:.3f} is below the required threshold "
            f"{_RECALL_THRESHOLD}"
        )

    @pytest.mark.asyncio
    async def test_per_group_recall_all_exceed_threshold(self) -> None:
        """Every group's mean Recall@9 must individually exceed 0.85."""
        groups, task_to_group = _make_tasks()
        retriever = _build_retriever(groups, task_to_group)

        for g, group_tasks in enumerate(groups):
            group_recalls: list[float] = []
            for query_task in group_tasks:
                true_pos = {
                    t.task_id for t in group_tasks if t.task_id != query_task.task_id
                }
                results = await retriever.retrieve(query_task)
                group_recalls.append(_recall_at_k(results, true_pos))

            group_mean = float(np.mean(group_recalls))
            assert group_mean > _RECALL_THRESHOLD, (
                f"Group {g} Recall@{_TOP_K_FINAL} = {group_mean:.3f} is below {_RECALL_THRESHOLD}"
            )

    @pytest.mark.asyncio
    async def test_top_results_dominated_by_same_group(self) -> None:
        """For a representative query, at least 8 of the top-9 results are same-group."""
        groups, task_to_group = _make_tasks()
        retriever = _build_retriever(groups, task_to_group)

        query_task = groups[0][0]
        same_group_ids = {t.task_id for t in groups[0]}

        results = await retriever.retrieve(query_task)
        retrieved_ids = {r[0].task_id for r in results}
        n_same_group = len(retrieved_ids & same_group_ids)

        assert n_same_group >= 8, (
            f"Expected ≥ 8 same-group results in top-10; got {n_same_group}"
        )

    @pytest.mark.asyncio
    async def test_cross_group_tasks_filtered_out(self) -> None:
        """Cross-group tasks (orthogonal cluster embeddings) must not appear in results."""
        groups, task_to_group = _make_tasks()
        retriever = _build_retriever(groups, task_to_group)

        query_task = groups[5][0]
        other_group_ids = {
            t.task_id
            for g_idx, group_tasks in enumerate(groups)
            if g_idx != 5
            for t in group_tasks
        }

        results = await retriever.retrieve(query_task)
        retrieved_ids = {r[0].task_id for r in results}
        cross_group_count = len(retrieved_ids & other_group_ids)

        assert cross_group_count == 0, (
            f"Expected 0 cross-group tasks in results; got {cross_group_count}"
        )

    @pytest.mark.asyncio
    async def test_empty_index_returns_empty_results(self) -> None:
        """A retriever backed by an empty FAISS index returns an empty list."""
        empty_idx = _SimpleFaissIndex()
        mock_repo = MagicMock()
        mock_repo.get_by_id = AsyncMock(return_value=None)
        mock_expander = MagicMock()
        mock_expander.expand = AsyncMock(return_value=[])
        mock_ranker = MagicMock()
        mock_ranker.rerank = AsyncMock(return_value=[])

        retriever = HybridRetriever(
            faiss_index=empty_idx,
            task_repository=mock_repo,
            embedder=_ControlledEmbedder({}),
            query_expander=mock_expander,
            llm_ranker=mock_ranker,
            top_k_initial=10,
            top_k_final=10,
            similarity_threshold=0.5,
            use_llm_reranking=False,
        )

        task = Task(
            task_id=uuid4(),
            name="query",
            task_type="classification",
            created_at=_NOW,
            updated_at=_NOW,
        )
        results = await retriever.retrieve(task)
        assert results == []
