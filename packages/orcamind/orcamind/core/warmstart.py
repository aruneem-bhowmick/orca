"""Warm-start transfer: initialise a new task's model from a similar historical checkpoint."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

import numpy as np
import torch.nn as nn

from orcamind.embedders.similarity import FaissIndex

if TYPE_CHECKING:
    from orca_shared.registry.repository import TaskRepository
    from orca_shared.schemas.task import Task
    from orca_shared.tracking.artifacts import ArtifactManager

logger = logging.getLogger(__name__)

_ENCODER_KEYWORDS: tuple[str, ...] = ("encoder", "backbone", "feature")
_HEAD_KEYWORDS: tuple[str, ...] = ("head", "classifier", "output")
_VALID_STRATEGIES: frozenset[str] = frozenset({"all", "encoder_only", "head_only"})


def _name_matches_keywords(name: str, keywords: tuple[str, ...]) -> bool:
    """Return True if any dot-separated segment of *name* exactly equals a keyword.

    Segment-level matching prevents ``"pre_encoder.weight"`` from being claimed
    by the ``"encoder"`` keyword — only ``"encoder.weight"`` (or deeper paths
    like ``"backbone.layer1.weight"``) will match.
    """
    segments = set(name.split("."))
    return bool(segments.intersection(keywords))


class WarmStartTransfer:
    """Transfer weights from a similar historical task to warm-start a new model."""

    def __init__(
        self,
        similarity_index: FaissIndex,
        artifact_manager: ArtifactManager,
        task_repository: TaskRepository,
        layer_selection: str = "all",
    ) -> None:
        self._similarity_index = similarity_index
        self._artifact_manager = artifact_manager
        self._task_repository = task_repository
        self._layer_selection = layer_selection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_source_task(
        self,
        target_embedding: np.ndarray,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """Return top-k (task_id, score) pairs from the similarity index."""
        return self._similarity_index.search(target_embedding, k=k)

    def transfer_weights(
        self,
        source_model: nn.Module,
        target_model: nn.Module,
        strategy: str = "all",
    ) -> nn.Module:
        """Copy parameters from *source_model* into *target_model* per *strategy*.

        Strategies:
            "all"          — copy all parameters whose names match between models
            "encoder_only" — copy only params whose name contains encoder/backbone/feature
            "head_only"    — copy only params whose name contains head/classifier/output

        Parameters with mismatched shapes are silently skipped (warning logged).
        Returns *target_model* in-place.

        Raises:
            ValueError: if *strategy* is not one of the accepted values.
        """
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. Valid strategies: {sorted(_VALID_STRATEGIES)}"
            )
        source_params = dict(source_model.named_parameters())
        for name, target_param in target_model.named_parameters():
            if name not in source_params:
                continue
            if strategy == "encoder_only" and not _name_matches_keywords(name, _ENCODER_KEYWORDS):
                continue
            if strategy == "head_only" and not _name_matches_keywords(name, _HEAD_KEYWORDS):
                continue
            source_param = source_params[name]
            if source_param.shape != target_param.shape:
                logger.warning(
                    "Skipping %s: shape mismatch %s vs %s",
                    name,
                    source_param.shape,
                    target_param.shape,
                )
                continue
            target_param.data.copy_(source_param.data)
        return target_model

    def get_adaptive_schedule(
        self,
        source_task: Task,
        target_task: Task,
        similarity_score: float | None = None,
    ) -> dict:
        """Return a training schedule dict calibrated to the source→target similarity.

        If *similarity_score* is not provided it is derived from task metadata fields.
        """
        if similarity_score is None:
            similarity_score = self._metadata_similarity(source_task, target_task)
        if similarity_score > 0.9:
            return {"lr_multiplier": 0.1, "freeze_backbone_epochs": 5}
        if similarity_score >= 0.6:
            return {"lr_multiplier": 0.3, "freeze_backbone_epochs": 2}
        return {"lr_multiplier": 1.0, "freeze_backbone_epochs": 0}

    async def warm_start(
        self,
        target_task_id: str,
        target_model: nn.Module,
        target_embedding: np.ndarray,
    ) -> tuple[nn.Module, dict]:
        """Orchestrate: find source → download checkpoint → transfer → schedule.

        Returns *(initialized_model, schedule_dict)*.  If no similar tasks are
        found the original *target_model* is returned with the default schedule.
        """
        candidates = self.find_source_task(target_embedding)
        if not candidates:
            logger.warning(
                "No source tasks found for %s; returning model unchanged with default schedule.",
                target_task_id,
            )
            return target_model, {"lr_multiplier": 1.0, "freeze_backbone_epochs": 0}

        source_task_id, score = candidates[0]
        try:
            source_uuid = uuid.UUID(source_task_id)
        except ValueError as exc:
            raise ValueError(
                f"Invalid source task UUID {source_task_id!r}: {exc}"
            ) from exc
        try:
            target_uuid = uuid.UUID(target_task_id)
        except ValueError as exc:
            raise ValueError(
                f"Invalid target task UUID {target_task_id!r}: {exc}"
            ) from exc
        source_task = await self._task_repository.get_by_id(source_uuid)
        target_task = await self._task_repository.get_by_id(target_uuid)
        checkpoint_uri: str = (source_task.metadata or {}).get(
            "checkpoint_uri",
            f"models/{source_task_id}/checkpoint",
        )
        source_model = await self._artifact_manager.download_model(checkpoint_uri)
        initialized_model = self.transfer_weights(source_model, target_model, self._layer_selection)
        schedule = self.get_adaptive_schedule(source_task, target_task, score)
        return initialized_model, schedule

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _metadata_similarity(source: Task, target: Task) -> float:
        """Heuristic similarity score derived from task metadata fields."""
        scores: list[float] = []
        for field in ("n_samples", "n_features", "n_classes"):
            s = getattr(source, field, None)
            t = getattr(target, field, None)
            if s is not None and t is not None and max(s, t) > 0:
                scores.append(min(s, t) / max(s, t))
        scores.append(1.0 if source.task_type == target.task_type else 0.0)
        return float(np.mean(scores))
