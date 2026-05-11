"""Training loops, Lightning modules, and experiment orchestration."""

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
    "UniformTaskSampler",
    "CurriculumTaskSampler",
    "DomainBalancedSampler",
    "k_shot_accuracy",
    "adaptation_efficiency",
    "catastrophic_forgetting",
]

# pytorch-lightning is an optional dependency; guard against missing install
try:
    from orcamind.training.callbacks import (
        CheckpointCallback,
        EarlyStoppingCallback,
        MetaValidationCallback,
    )
    from orcamind.training.meta_trainer import MetaTrainer

    __all__ += [
        "MetaValidationCallback",
        "EarlyStoppingCallback",
        "CheckpointCallback",
        "MetaTrainer",
    ]
except ImportError:
    pass
