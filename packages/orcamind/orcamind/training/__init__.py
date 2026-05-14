"""Training loops, Lightning modules, and experiment orchestration."""

_import_error = None

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
    _import_error = _err

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

_OPTIONAL_NAMES = frozenset(__all__)


def __getattr__(name: str) -> object:
    if name in _OPTIONAL_NAMES and _import_error is not None:
        raise ImportError(
            f"Cannot import '{name}' from 'orcamind.training': {_import_error}. "
            "Install pytorch-lightning to use the training pipeline."
        ) from _import_error
    raise AttributeError(f"module 'orcamind.training' has no attribute {name!r}")
