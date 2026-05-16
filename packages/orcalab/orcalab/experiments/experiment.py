"""Experiment dataclass extending the shared ExperimentResult schema."""

from __future__ import annotations

from typing import Any

from orca_shared.schemas.training import ExperimentResult, TrainingConfig


class Experiment(ExperimentResult):
    """A fully-specified experiment ready for execution.

    Extends ExperimentResult with the architecture config, training
    hyperparameters, and free-form tags needed to actually run a trial.

    Note: the field is named ``arch_config`` rather than ``model_config``
    because ``model_config`` is a reserved class-level attribute in Pydantic v2.
    """

    arch_config: dict[str, Any] | None = None
    training_config: TrainingConfig | None = None
    tags: dict[str, str] | None = None
