"""Training loops, Lightning modules, and experiment orchestration."""

import warnings as _warnings

try:
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
except ImportError as _err:
    _warnings.warn(
        f"orcamind.training requires optional ML dependencies ({_err}). "
        "Install pytorch-lightning to use the training pipeline.",
        ImportWarning,
        stacklevel=2,
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
