"""Three-way integration pipeline: OrcaNet ↔ OrcaMind ↔ OrcaLab."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import httpx
from pydantic import BaseModel

from orca_shared.clients.orcalab_client import OrcaLabClient
from orca_shared.clients.orcamind_client import OrcaMindClient
from orca_shared.registry.repository import TaskRepository
from orca_shared.schemas.training import ExperimentResult
from orca_shared.schemas.transfer import TransferMapping
from orcanet.transfer.base import TransferStrategy
from orcanet.transfer.types import TransferScore

logger = logging.getLogger("orcanet.integration")


class ServiceUnavailableError(Exception):
    """Raised when a required upstream service (OrcaMind or OrcaLab) is unreachable."""


class TransferValidationResult(BaseModel):
    """Result of a full transfer pipeline run including optional OrcaLab validation."""

    score: TransferScore
    experiment_result: ExperimentResult | None
    mapping: TransferMapping
    improvement_over_baseline: float | None = None

    model_config = {"arbitrary_types_allowed": True}


class TransferPipeline:
    """Coordinates OrcaNet transfer scoring with OrcaMind model retrieval and OrcaLab validation.

    Workflow for ``recommend_and_validate``:

    1. Retrieve the best source model from OrcaMind.
    2. Score the knowledge transfer between source and target tasks.
    3. Optionally submit a validation experiment to OrcaLab (score threshold: > 0.4).
    4. Persist the transfer mapping to the database.
    5. Return a ``TransferValidationResult`` with all results.
    """

    def __init__(
        self,
        orcamind_client: OrcaMindClient,
        orcalab_client: OrcaLabClient,
        transfer_strategies: dict[str, TransferStrategy],
        task_repository: TaskRepository,
    ) -> None:
        """Initialise the pipeline with its three upstream service clients.

        Args:
            orcamind_client: Client for the OrcaMind meta-learning service.
            orcalab_client: Client for the OrcaLab experiment orchestration service.
            transfer_strategies: Mapping of strategy name to :class:`TransferStrategy`
                instance (e.g. ``{"feature": FeatureTransfer(...)}``.
            task_repository: Repository for reading tasks and persisting mappings.
        """
        self._orcamind = orcamind_client
        self._orcalab = orcalab_client
        self._strategies = transfer_strategies
        self._repo = task_repository

    async def recommend_and_validate(
        self,
        source_task_id: str,
        target_task_id: str,
        strategy_name: str = "feature",
        validate: bool = True,
    ) -> TransferValidationResult:
        """Run the full transfer pipeline for the given source/target task pair.

        Args:
            source_task_id: UUID string of the source (donor) task.
            target_task_id: UUID string of the target (recipient) task.
            strategy_name: Transfer strategy key — one of ``feature``, ``weight``,
                ``architecture``, or any key registered in *transfer_strategies*.
            validate: When ``True`` and the transfer score exceeds 0.4, submit an
                OrcaLab experiment to empirically validate the transfer.

        Returns:
            A :class:`TransferValidationResult` containing the transfer score,
            optional experiment result, persisted mapping, and improvement estimate.

        Raises:
            ServiceUnavailableError: When OrcaMind cannot be reached.
            KeyError: When *strategy_name* is not registered.
            ValueError: When either task ID is not found in the repository.

        Note:
            If OrcaLab validation times out, the mapping is still saved and
            ``experiment_result`` will be ``None`` (no exception is raised).
        """
        source_uuid = UUID(source_task_id)
        target_uuid = UUID(target_task_id)

        # Step 1: Retrieve the best source model from OrcaMind.
        try:
            source_model_config = await self._orcamind.get_best_model(source_uuid)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise ServiceUnavailableError(
                f"OrcaMind is unreachable: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                raise ServiceUnavailableError(
                    f"OrcaMind returned {exc.response.status_code}: {exc}"
                ) from exc
            raise

        # Step 2: Score the transfer.
        source_task = await self._repo.get_by_id(source_uuid)
        if source_task is None:
            raise ValueError(f"Source task {source_task_id!r} not found")
        target_task = await self._repo.get_by_id(target_uuid)
        if target_task is None:
            raise ValueError(f"Target task {target_task_id!r} not found")

        strategy = self._strategies[strategy_name]
        score: TransferScore = strategy.score_transfer(source_task, target_task)

        # Step 3 (optional): Submit a validation experiment to OrcaLab.
        experiment_result: ExperimentResult | None = None
        experiment_id: str | None = None

        if validate and score.overall > 0.4:
            try:
                experiment_id = await self._orcalab.create_experiment(
                    task_id=target_task_id,
                    model_config=source_model_config,
                    tags=[
                        "transfer_validation",
                        f"source:{source_task_id}",
                        f"strategy:{strategy_name}",
                    ],
                )
                experiment_result = await self._orcalab.wait_for_completion(
                    experiment_id, timeout=3600
                )
            except TimeoutError:
                logger.warning(
                    "OrcaLab validation timed out for experiment %s; "
                    "mapping will be saved without validated_performance",
                    experiment_id,
                )
                experiment_result = None

        # Step 4: Persist the transfer mapping.
        validated_performance: float | None = None
        if experiment_result is not None and experiment_result.metrics:
            validated_performance = experiment_result.metrics.get("accuracy")

        mapping = await self._repo.save_transfer_mapping(
            source_task_id=source_uuid,
            target_task_id=target_uuid,
            transfer_score=score.overall,
            transfer_type=strategy_name,
            metadata={
                "layer_scores": score.layer_scores,
                "experiment_id": str(experiment_id) if experiment_id else None,
                "validated_performance": validated_performance,
            },
        )

        # Derive improvement over baseline if the experiment recorded both values.
        improvement: float | None = None
        if experiment_result is not None and experiment_result.metrics:
            baseline = experiment_result.metrics.get("baseline_accuracy")
            if baseline is not None and validated_performance is not None:
                improvement = validated_performance - baseline

        return TransferValidationResult(
            score=score,
            experiment_result=experiment_result,
            mapping=mapping,
            improvement_over_baseline=improvement,
        )
