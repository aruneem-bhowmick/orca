"""Three-stage hybrid task retriever: FAISS vector search → metadata filter → LLM re-rank."""

from __future__ import annotations

import asyncio
import logging
import math
from uuid import UUID

import numpy as np
import torch

from orca_shared.registry.repository import TaskRepository
from orca_shared.schemas.task import Task
from orcanet.embeddings.cross_domain import CrossDomainEmbedder
from orcanet.retrieval.query_expander import QueryExpander
from orcanet.retrieval.ranker import LLMRanker

logger = logging.getLogger(__name__)

_FEATURE_DIM = 25  # must match CrossDomainEmbedder default input_dim


def _task_to_feature_vector(task: Task) -> np.ndarray:
    """Build a 25-dim float32 feature vector from a Task's statistical fields."""
    vec = np.zeros(_FEATURE_DIM, dtype=np.float32)
    if task.n_samples is not None:
        vec[0] = math.log1p(float(task.n_samples))
    if task.n_features is not None:
        vec[1] = float(task.n_features)
    if task.n_classes is not None:
        vec[2] = float(task.n_classes)
    return vec


def _deduplicate_and_sort(
    results: list[tuple[Task, float, str]],
) -> list[tuple[Task, float, str]]:
    """Keep highest-scoring entry per task_id and return sorted by score descending."""
    best: dict[UUID, tuple[Task, float, str]] = {}
    for task, score, reasoning in results:
        if task.task_id not in best or score > best[task.task_id][1]:
            best[task.task_id] = (task, score, reasoning)
    return sorted(best.values(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """Three-stage retrieval: FAISS vector search → metadata filter → LLM re-rank."""

    def __init__(
        self,
        faiss_index,
        task_repository: TaskRepository,
        embedder: CrossDomainEmbedder,
        query_expander: QueryExpander,
        llm_ranker: LLMRanker,
        *,
        top_k_initial: int = 50,
        top_k_final: int = 10,
        similarity_threshold: float = 0.6,
        use_llm_reranking: bool = True,
    ) -> None:
        self._index = faiss_index
        self._repo = task_repository
        self._embedder = embedder
        self._expander = query_expander
        self._ranker = llm_ranker
        self.top_k_initial = top_k_initial
        self.top_k_final = top_k_final
        self.similarity_threshold = similarity_threshold
        self.use_llm_reranking = use_llm_reranking

    async def retrieve(
        self,
        query_task: Task,
        filters: dict | None = None,
    ) -> list[tuple[Task, float, str]]:
        """Return up to *top_k_final* (task, score, reasoning) tuples.

        Stage 1 — FAISS vector similarity; Stage 2 — metadata filter + threshold;
        Stage 3 — optional LLM re-rank.
        """
        # Stage 1: FAISS vector similarity
        feature_vec = _task_to_feature_vector(query_task)
        query_tensor = torch.from_numpy(feature_vec).unsqueeze(0)
        embedding = self._embedder.embed(query_tensor).squeeze(0).detach().numpy()
        candidate_ids: list[tuple[str, float]] = self._index.search(
            embedding, k=self.top_k_initial
        )
        if not candidate_ids:
            return []

        score_by_id = {task_id: score for task_id, score in candidate_ids}

        # Stage 2: Batch fetch + threshold filter + metadata filter
        fetched = await asyncio.gather(
            *[self._repo.get_by_id(UUID(tid)) for tid in score_by_id],
            return_exceptions=True,
        )
        candidates: list[Task] = []
        for item in fetched:
            if isinstance(item, Exception):
                logger.warning("HybridRetriever: failed to fetch candidate task", exc_info=item)
                continue
            if item is not None:
                candidates.append(item)
        candidates = [
            t for t in candidates
            if score_by_id.get(str(t.task_id), 0.0) >= self.similarity_threshold
        ]
        if filters:
            for key, val in filters.items():
                candidates = [t for t in candidates if getattr(t, key, None) == val]

        if not candidates:
            return []

        # Stage 3: LLM re-ranking (optional)
        if self.use_llm_reranking and len(candidates) > self.top_k_final:
            return await self._ranker.rerank(query_task, candidates, top_k=self.top_k_final)

        return [
            (t, score_by_id.get(str(t.task_id), 0.0), "vector similarity")
            for t in candidates[: self.top_k_final]
        ]

    async def retrieve_with_expanded_queries(
        self,
        query_description: str,
        query_task: Task,
    ) -> list[tuple[Task, float, str]]:
        """Expand *query_description* and aggregate results across all expansions."""
        expansions = await self._expander.expand(query_description)
        all_results: list[tuple[Task, float, str]] = []
        for description in [query_description, *expansions]:
            query_variant = query_task.model_copy(update={"name": description})
            results = await self.retrieve(query_variant)
            all_results.extend(results)
        return _deduplicate_and_sort(all_results)[: self.top_k_final]
