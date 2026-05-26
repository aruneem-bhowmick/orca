"""Unit tests for HybridRetriever, _task_to_feature_vector, and _deduplicate_and_sort."""

from __future__ import annotations

import math
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import torch

from orca_shared.schemas.task import Task
from orcanet.retrieval.retriever import (
    HybridRetriever,
    _deduplicate_and_sort,
    _task_to_feature_vector,
)

_NOW = datetime(2024, 1, 1)


def _make_task(**overrides) -> Task:
    defaults = dict(
        task_id=uuid4(),
        name="test-task",
        domain="vision",
        task_type="classification",
        n_samples=1000,
        n_features=10,
        n_classes=3,
        created_at=_NOW,
        updated_at=_NOW,
    )
    defaults.update(overrides)
    return Task(**defaults)


def _make_mocks(task: Task) -> SimpleNamespace:
    """Return pre-configured mocks for a single-task happy-path scenario."""
    index = MagicMock()
    index.search.return_value = [(str(task.task_id), 0.9)]

    repo = MagicMock()
    repo.get_by_id = AsyncMock(return_value=task)

    embedder = MagicMock()
    embedder.embed.return_value = torch.zeros(25)

    expander = MagicMock()
    expander.expand = AsyncMock(return_value=["alt one", "alt two"])

    ranker = MagicMock()
    ranker.rerank = AsyncMock(return_value=[(task, 0.95, "high relevance")])

    return SimpleNamespace(
        index=index,
        repo=repo,
        embedder=embedder,
        expander=expander,
        ranker=ranker,
    )


def _build_retriever(mocks: SimpleNamespace, **cfg_overrides) -> HybridRetriever:
    cfg = dict(
        top_k_initial=50,
        top_k_final=10,
        similarity_threshold=0.6,
        use_llm_reranking=True,
    )
    cfg.update(cfg_overrides)
    return HybridRetriever(
        faiss_index=mocks.index,
        task_repository=mocks.repo,
        embedder=mocks.embedder,
        query_expander=mocks.expander,
        llm_ranker=mocks.ranker,
        **cfg,
    )


# ---------------------------------------------------------------------------
# _task_to_feature_vector
# ---------------------------------------------------------------------------


class TestTaskToFeatureVector:
    def test_returns_25_dim_float32_vector(self) -> None:
        vec = _task_to_feature_vector(_make_task())
        assert vec.shape == (25,)
        assert vec.dtype.name == "float32"

    def test_n_samples_encoded_as_log1p(self) -> None:
        vec = _task_to_feature_vector(_make_task(n_samples=1000))
        assert vec[0] == pytest.approx(math.log1p(1000))

    def test_none_fields_produce_zeros(self) -> None:
        vec = _task_to_feature_vector(
            _make_task(n_samples=None, n_features=None, n_classes=None)
        )
        assert (vec == 0.0).all()


# ---------------------------------------------------------------------------
# _deduplicate_and_sort
# ---------------------------------------------------------------------------


class TestDeduplicateAndSort:
    def test_keeps_highest_score_per_task_id(self) -> None:
        task = _make_task()
        results = [
            (task, 0.7, "first"),
            (task, 0.9, "second"),
            (task, 0.5, "third"),
        ]
        deduped = _deduplicate_and_sort(results)
        assert len(deduped) == 1
        assert deduped[0][1] == pytest.approx(0.9)

    def test_returns_descending_order(self) -> None:
        tasks = [_make_task(name=f"t{i}") for i in range(3)]
        results = [
            (tasks[0], 0.5, "mid"),
            (tasks[1], 0.9, "high"),
            (tasks[2], 0.3, "low"),
        ]
        deduped = _deduplicate_and_sort(results)
        scores = [r[1] for r in deduped]
        assert scores == sorted(scores, reverse=True)

    def test_single_entry_returned_unchanged(self) -> None:
        task = _make_task()
        result = _deduplicate_and_sort([(task, 0.8, "only")])
        assert len(result) == 1
        assert result[0][0] is task
        assert result[0][1] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    @pytest.mark.asyncio
    async def test_retrieve_returns_empty_when_faiss_returns_nothing(self) -> None:
        task = _make_task()
        mocks = _make_mocks(task)
        mocks.index.search.return_value = []
        retriever = _build_retriever(mocks)
        result = await retriever.retrieve(task)
        assert result == []

    @pytest.mark.asyncio
    async def test_similarity_threshold_filters_low_score_candidates(self) -> None:
        high_task = _make_task(name="high-score")
        low_task = _make_task(name="low-score")
        task_map = {high_task.task_id: high_task, low_task.task_id: low_task}

        mocks = _make_mocks(high_task)
        mocks.index.search.return_value = [
            (str(high_task.task_id), 0.9),
            (str(low_task.task_id), 0.4),
        ]
        mocks.repo.get_by_id = AsyncMock(side_effect=lambda uid: task_map.get(uid))

        retriever = _build_retriever(mocks, similarity_threshold=0.6, use_llm_reranking=False)
        result = await retriever.retrieve(high_task)

        task_ids = {r[0].task_id for r in result}
        assert low_task.task_id not in task_ids
        assert high_task.task_id in task_ids

    @pytest.mark.asyncio
    async def test_use_llm_reranking_false_skips_ranker(self) -> None:
        task = _make_task()
        mocks = _make_mocks(task)
        retriever = _build_retriever(mocks, use_llm_reranking=False)
        await retriever.retrieve(task)
        mocks.ranker.rerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_reranking_triggered_when_candidates_exceed_top_k_final(
        self,
    ) -> None:
        tasks = [_make_task(name=f"t{i}") for i in range(3)]
        task_map = {t.task_id: t for t in tasks}

        mocks = _make_mocks(tasks[0])
        mocks.index.search.return_value = [
            (str(t.task_id), 0.9 - 0.05 * i) for i, t in enumerate(tasks)
        ]
        mocks.repo.get_by_id = AsyncMock(side_effect=lambda uid: task_map.get(uid))
        mocks.ranker.rerank = AsyncMock(
            return_value=[(tasks[0], 0.95, "best"), (tasks[1], 0.85, "second")]
        )

        retriever = _build_retriever(
            mocks,
            similarity_threshold=0.7,
            top_k_final=2,
            use_llm_reranking=True,
        )
        await retriever.retrieve(tasks[0])
        mocks.ranker.rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_metadata_filter_removes_non_matching_tasks(self) -> None:
        task = _make_task(domain="vision")
        mocks = _make_mocks(task)
        retriever = _build_retriever(mocks, use_llm_reranking=False)
        result = await retriever.retrieve(task, filters={"domain": "nlp"})
        assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_with_expanded_queries_deduplicates_by_task_id(
        self,
    ) -> None:
        task = _make_task()
        mocks = _make_mocks(task)
        mocks.expander.expand = AsyncMock(return_value=["alt one", "alt two"])
        retriever = _build_retriever(mocks, use_llm_reranking=False)

        result = await retriever.retrieve_with_expanded_queries(
            "brain MRI classification", task
        )
        task_ids = [r[0].task_id for r in result]
        assert len(task_ids) == len(set(task_ids))
