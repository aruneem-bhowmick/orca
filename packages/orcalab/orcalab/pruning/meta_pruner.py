"""MetaPruner — OrcaMind-guided pruner with base_pruner fallback."""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from orca_shared.clients.orcamind_client import OrcaMindClient
from orcalab.pruning.base import Pruner

logger = logging.getLogger(__name__)


class MetaPruner(Pruner):
    """Uses OrcaMind performance predictions for early-stopping decisions.

    Queries OrcaMind's PerformancePredictor using the trial's observed value
    history as the task embedding.  If the predicted final performance falls
    below ``prediction_threshold`` the trial is pruned immediately.

    When fewer than ``min_steps_before_prediction`` steps have been observed,
    or when OrcaMind is unavailable, the decision is delegated to
    ``base_pruner`` so the sweep is never blocked by a network error.
    """

    def __init__(
        self,
        orcamind_client: OrcaMindClient,
        base_pruner: Pruner,
        prediction_threshold: float = 0.3,
        min_steps_before_prediction: int = 10,
    ) -> None:
        self._client = orcamind_client
        self._base_pruner = base_pruner
        self._threshold = prediction_threshold
        self._min_steps = min_steps_before_prediction

    @property
    def name(self) -> str:
        return "meta_pruner"

    def should_prune(
        self,
        trial_id: str,
        step: int,
        current_value: float,
        all_trial_values: dict[str, list[float]],
    ) -> bool:
        if step < self._min_steps:
            return False

        predicted = self._query_orcamind(trial_id, step, current_value, all_trial_values)
        if predicted is not None and predicted < self._threshold:
            logger.debug(
                "MetaPruner: pruning trial %r at step %d (predicted=%.4f < threshold=%.4f)",
                trial_id,
                step,
                predicted,
                self._threshold,
            )
            return True

        return self._base_pruner.should_prune(trial_id, step, current_value, all_trial_values)

    def _query_orcamind(
        self,
        trial_id: str,
        step: int,
        current_value: float,
        all_trial_values: dict[str, list[float]],
    ) -> float | None:
        """Synchronously query OrcaMind for the predicted final performance.

        Uses the trial's observed values as the task embedding vector.
        Returns None on any error so the caller falls through to base_pruner.
        """
        try:
            curve = all_trial_values.get(trial_id, []) + [current_value]
            loop = asyncio.new_event_loop()
            try:
                metrics = loop.run_until_complete(
                    self._client.predict_performance(
                        task_embedding=[float(v) for v in curve],
                        model_id=uuid4(),
                    )
                )
            finally:
                loop.close()
            if metrics.final_metrics:
                return max(metrics.final_metrics.values())
            return None
        except Exception as exc:
            logger.warning(
                "OrcaMind prediction failed for trial %r at step %d: %s",
                trial_id,
                step,
                exc,
            )
            return None
