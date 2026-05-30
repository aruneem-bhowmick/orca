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


def task_to_feature_vector(task: Task) -> np.ndarray:
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
        """Initialise the retriever with its three-stage pipeline components.

        Args:
            faiss_index:         FAISS (or compatible) index with a ``search(embedding, k)``
                                 method that returns ``[(task_id_str, score), ...]``.
            task_repository:     Repository used to hydrate task metadata from IDs.
            embedder:            ``CrossDomainEmbedder`` used to project the query
                                 task's statistical feature vector into 64-dim space.
            query_expander:      ``QueryExpander`` that generates alternative
                                 phrasings for expanded-query retrieval.
            llm_ranker:          ``LLMRanker`` that re-orders candidates in Stage 3.
            top_k_initial:       Number of FAISS candidates retrieved in Stage 1.
            top_k_final:         Maximum results returned to the caller.
            similarity_threshold: Minimum FAISS cosine score to retain a candidate
                                 after Stage 1 (Stage 2 filter).
            use_llm_reranking:   When ``True`` and ``len(candidates) > top_k_final``,
                                 Stage 3 LLM re-ranking is applied.
        """
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
        task_repository: TaskRepository | None = None,
    ) -> list[tuple[Task, float, str]]:
        """Return up to *top_k_final* (task, score, reasoning) tuples.

        Stage 1 — FAISS vector similarity; Stage 2 — metadata filter + threshold;
        Stage 3 — optional LLM re-rank.

        ``task_repository`` overrides the instance-level repo so a request-scoped
        session can be used without mutating shared state.
        """
        repo = task_repository if task_repository is not None else self._repo
        # Stage 1: FAISS vector similarity
        feature_vec = task_to_feature_vector(query_task)
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
            *[repo.get_by_id(UUID(tid)) for tid in score_by_id],
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
        task_repository: TaskRepository | None = None,
    ) -> list[tuple[Task, float, str]]:
        """Expand *query_description* and aggregate results across all expansions.

        Stage 1 (FAISS vector similarity) is driven exclusively by the statistical
        features of *query_task* (``n_samples``, ``n_features``, ``n_classes``).
        Because these fields are unchanged across expansions, Stage 1 returns an
        identical candidate set for every phrasing.  The fan-out benefit therefore
        applies to Stage 3 (LLM re-ranking, when ``use_llm_reranking=True``): each
        expanded variant is passed as the query phrasing so the ranker sees a richer
        description of the retrieval intent.  ``_deduplicate_and_sort`` collapses
        the repeated Stage 1 results to a single sorted list.

        For text-only queries where *query_task* carries ``None`` statistical fields,
        Stage 1 searches with a zero-vector embedding; Stage 3 is then the sole
        source of ranking signal.
        """
        expansions = await self._expander.expand(query_description)
        all_results: list[tuple[Task, float, str]] = []
        for description in [query_description, *expansions]:
            query_variant = query_task.model_copy(update={"name": description})
            results = await self.retrieve(query_variant, task_repository=task_repository)
            all_results.extend(results)
        return _deduplicate_and_sort(all_results)[: self.top_k_final]
