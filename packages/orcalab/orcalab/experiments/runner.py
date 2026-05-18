"""Single-experiment runner with retry logic and pruner integration."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Protocol
from uuid import UUID

from orca_shared.schemas.training import ExperimentResult
from orca_shared.tracking.artifacts import ArtifactManager
from orca_shared.tracking.tracker import OrcaTracker

from orcalab.experiments.experiment import Experiment
from orcalab.experiments.lifecycle import ExperimentLifecycle, ExperimentStatus
from orcalab.pruning.base import Pruner


class TrainableModel(Protocol):
    """Minimal interface a model must expose for the runner's training loop."""

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch and return the primary metric (higher = better)."""
        ...


class _NullRepository:
    """No-op repository used when the runner has no backing database."""

    async def update_status(self, experiment_id: UUID, status: str) -> None:  # noqa: ARG002
        return

    async def update_status_if_current(  # noqa: ARG002
        self, experiment_id: UUID, from_status: str, to_status: str
    ) -> bool:
        return True

    async def update_metrics(  # noqa: ARG002
        self, experiment_id: UUID, metrics: dict[str, Any]
    ) -> None:
        return


def _build_result(
    experiment: Experiment,
    mlflow_run_id: str | None,
    status: str,
    metrics: dict[str, float] | None = None,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment.experiment_id,
        task_id=experiment.task_id,
        model_id=experiment.model_id,
        status=status,
        mlflow_run_id=mlflow_run_id,
        started_at=experiment.started_at,
        completed_at=experiment.completed_at,
        metrics=metrics,
    )


class ExperimentRunner:
    """Executes a single experiment: trains a model, logs to MLflow, retries on failure.

    Retries stay within the RUNNING state — only the final outcome triggers a
    COMPLETED or FAILED transition.  A pruner can cut a trial short at any epoch
    boundary, in which case the experiment is immediately marked FAILED with
    reason ``"pruned"``.
    """

    def __init__(
        self,
        tracker: OrcaTracker,
        artifact_manager: ArtifactManager,
        max_retries: int = 2,
        timeout: int = 3600,
        model_factory: Callable[[dict[str, Any]], TrainableModel] | None = None,
        repository: Any | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        self._tracker = tracker
        self._artifact_manager = artifact_manager
        self._max_retries = max_retries
        self._timeout = timeout
        self._model_factory = model_factory or self._no_factory
        self._repository = repository if repository is not None else _NullRepository()

    @staticmethod
    def _no_factory(_config: dict[str, Any]) -> TrainableModel:
        raise NotImplementedError(
            "No model_factory provided to ExperimentRunner. "
            "Pass a callable that accepts arch_config and returns a TrainableModel."
        )

    async def run(
        self,
        experiment: Experiment,
        pruner: Pruner | None = None,
    ) -> ExperimentResult:
        """Train *experiment*, streaming metrics to MLflow, with up to *max_retries* retries.

        Returns an ExperimentResult whose status is "completed", "failed", or
        "failed" (reason="pruned") depending on the outcome.
        """
        lifecycle = ExperimentLifecycle(experiment, self._repository)  # type: ignore[arg-type]
        await lifecycle.transition(ExperimentStatus.RUNNING)

        epochs = experiment.training_config.epochs if experiment.training_config else 10
        trial_id = str(experiment.experiment_id)

        last_exc: BaseException | None = None

        for _attempt in range(self._max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self._attempt(experiment, lifecycle, pruner, epochs, trial_id),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError as exc:
                last_exc = exc
            except _PrunedError:
                return _build_result(experiment, None, "failed")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc

        await lifecycle.transition(ExperimentStatus.FAILED, reason=str(last_exc))
        return _build_result(experiment, None, "failed")

    async def _attempt(
        self,
        experiment: Experiment,
        lifecycle: ExperimentLifecycle,
        pruner: Pruner | None,
        epochs: int,
        trial_id: str,
    ) -> ExperimentResult:
        async with self._tracker as tracker:
            model = self._model_factory(experiment.arch_config or {})
            if experiment.training_config:
                tracker.log_params(experiment.training_config.model_dump(exclude_none=True))

            trial_values: list[float] = []
            final_metrics: dict[str, float] = {}

            for epoch in range(1, epochs + 1):
                metric = model.train_epoch(epoch)
                tracker.log_metric("loss", metric, step=epoch)
                trial_values.append(metric)
                live_metrics: dict[str, Any] = {"loss": metric, "epoch": epoch}
                final_metrics.update(live_metrics)
                await self._repository.update_metrics(experiment.experiment_id, live_metrics)

                if pruner is not None and pruner.should_prune(
                    trial_id,
                    epoch,
                    metric,
                    {trial_id: trial_values[:-1]},
                ):
                    await lifecycle.transition(ExperimentStatus.FAILED, reason="pruned")
                    raise _PrunedError

            await self._artifact_manager.upload_model(
                model, f"model_{experiment.experiment_id}"
            )
            await lifecycle.transition(ExperimentStatus.COMPLETED)
            return _build_result(
                experiment, tracker.run_id, "completed", metrics=final_metrics
            )


class _PrunedError(Exception):
    """Internal sentinel raised when a pruner halts a trial."""
