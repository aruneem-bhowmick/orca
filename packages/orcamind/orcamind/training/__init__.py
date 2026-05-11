"""Training loops, Lightning modules, and experiment orchestration."""

from orcamind.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    MetaValidationCallback,
)
from orcamind.training.meta_trainer import MetaTrainer
from orcamind.training.metrics import (
    adaptation_efficiency,
    catastrophic_forgetting,
    k_shot_accuracy,
)
from orcamind.training.task_sampler import (
    CurriculumTaskSampler,
    DomainBalancedSampler,
    UniformTaskSampler,
)

__all__ = [
    "CheckpointCallback",
    "CurriculumTaskSampler",
    "DomainBalancedSampler",
    "EarlyStoppingCallback",
    "MetaTrainer",
    "MetaValidationCallback",
    "UniformTaskSampler",
    "adaptation_efficiency",
    "catastrophic_forgetting",
    "k_shot_accuracy",
]
