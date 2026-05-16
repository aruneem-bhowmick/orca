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

        Fetches the task embedding, top model recommendation, and the most similar
        historical tasks. For each similar task, samples a random candidate
        hyperparameter config from ``search_space`` and scores it as the product of
        that task's similarity coefficient and the recommendation's predicted
        performance, scaled by ``prior_weight``. This gives each prior a distinct,
        task-informed score rather than a single shared value.

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
            for similar_task in similar_tasks[: self._top_k_priors]:
                trial = sampler_study.ask()
                params = search_space.sample(trial)
                score = similar_task.score * recommendation.predicted_score
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
        ``FeedbackRequest`` (including the trial's hyperparameter params) so
        OrcaMind's meta-dataset reflects both the outcome and the configuration
        that produced it.

        Network and HTTP errors per submission are caught and logged so the rest
        of the batch is still attempted. On a fully successful flush,
        ``_completed_results`` is cleared so a second call is a no-op.
        """
        total = len(self._completed_results)
        failed = 0
        for trial_params, result in self._completed_results:
            req = FeedbackRequest(
                experiment_id=uuid4(),
                actual_metric=result,
                metric_name="objective",
                params=trial_params,
            )
            try:
                await self._client.submit_feedback(req)
            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                logger.warning(
                    "Failed to submit feedback to OrcaMind for task %r: %s",
                    task_id,
                    exc,
                )
                failed += 1
        if failed == 0:
            self._completed_results.clear()
        logger.info(
            "Flushed %d/%d results to OrcaMind for task %r",
            total - failed,
            total,
            task_id,
        )
