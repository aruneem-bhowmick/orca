"""MetaInformedSearch — Bayesian search warm-started with OrcaMind priors."""

from __future__ import annotations

import logging
import math
from typing import Any
from uuid import UUID, uuid4

import httpx
import optuna
import optuna.samplers

from orca_shared.clients.orcamind_client import OrcaMindClient
from orca_shared.schemas.recommendation import FeedbackRequest, RecommendationRequest
from orcalab.search.base import SearchStrategy
from orcalab.search.bayesian import BayesianSearch
from orcalab.search_spaces.space import SearchSpace

logger = logging.getLogger(__name__)


class MetaInformedSearch(SearchStrategy):
    """Bayesian search strategy warm-started with priors from OrcaMind.

    Call ``initialize_from_orcamind`` once before the sweep loop to seed the
    wrapped ``BayesianSearch`` with historically-informed starting points.
    After the sweep, call ``flush_results_to_orcamind`` to send completed trial
    outcomes back so OrcaMind's meta-dataset stays current.

    If OrcaMind is unreachable during initialization the strategy falls back to
    a plain Bayesian search — the sweep is never blocked by a network error.

    Example usage::

        space = SearchSpace("resnet").add(FloatParameter("lr", 1e-5, 1e-1))
        strategy = MetaInformedSearch(orcamind_client=client)
        await strategy.initialize_from_orcamind(task_id, space)
        for _ in range(50):
            params = strategy.suggest(space)
            result = train(params)
            strategy.update(params, result)
        await strategy.flush_results_to_orcamind(task_id)
    """

    def __init__(
        self,
        orcamind_client: OrcaMindClient,
        base_strategy: BayesianSearch | None = None,
        prior_weight: float = 1.0,
        top_k_priors: int = 10,
    ) -> None:
        self._client = orcamind_client
        self._base = base_strategy or BayesianSearch()
        self._prior_weight = prior_weight
        self._top_k_priors = top_k_priors
        self._completed_results: list[tuple[dict[str, Any], float]] = []

    async def initialize_from_orcamind(
        self, task_id: str, search_space: SearchSpace
    ) -> None:
        """Warm-start the base Bayesian strategy with priors fetched from OrcaMind.

        Fetches the task embedding, top model recommendation, and similar tasks.
        For each similar task, samples a candidate hyperparameter config from
        ``search_space`` and pairs it with OrcaMind's predicted performance score.
        Falls back silently on any network or HTTP error so the sweep is never blocked.
        """
        try:
            embedding = await self._client.embed_task(UUID(task_id))
            task_vec = embedding.embedding_vector

            recommendation = await self._client.recommend_model(
                RecommendationRequest(
                    task_embedding=task_vec,
                    top_k=self._top_k_priors,
                )
            )

            similar_tasks = await self._client.find_similar_tasks(
                task_vec, top_k=self._top_k_priors
            )

            sampler_study = optuna.create_study(
                sampler=optuna.samplers.RandomSampler()
            )
            priors: list[tuple[dict[str, Any], float]] = []
            for _ in similar_tasks[: self._top_k_priors]:
                trial = sampler_study.ask()
                params = search_space.sample(trial)
                perf = await self._client.predict_performance(
                    task_vec, recommendation.model_id
                )
                score = perf.final_metrics.get(
                    "accuracy", recommendation.predicted_score
                )
                priors.append((params, score * self._prior_weight))

            if priors:
                self._base.inject_priors(priors, search_space)
                logger.info(
                    "Injected %d OrcaMind priors into base strategy", len(priors)
                )

        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.warning(
                "OrcaMind unreachable — falling back to uninformed Bayesian search: %s",
                exc,
            )
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "OrcaMind returned %d — falling back to uninformed Bayesian search: %s",
                exc.response.status_code,
                exc,
            )

    def suggest(self, search_space: SearchSpace) -> dict[str, Any]:
        return self._base.suggest(search_space)

    def update(self, params: dict[str, Any], result: float) -> None:
        self._base.update(params, result)
        if math.isfinite(result):
            self._completed_results.append((params, result))

    def get_best(self, n: int = 1) -> list[tuple[dict, float]]:
        return self._base.get_best(n)

    @property
    def n_trials(self) -> int:
        return self._base.n_trials

    async def flush_results_to_orcamind(self, task_id: str) -> None:
        """Submit all completed trial results to OrcaMind as feedback.

        Each finite-result trial recorded via ``update`` is sent as a
        ``FeedbackRequest`` so OrcaMind's meta-dataset reflects the sweep outcomes.
        """
        for _params, result in self._completed_results:
            req = FeedbackRequest(
                experiment_id=uuid4(),
                actual_metric=result,
                metric_name="objective",
            )
            await self._client.submit_feedback(req)
        logger.info(
            "Flushed %d results to OrcaMind for task %r",
            len(self._completed_results),
            task_id,
        )
