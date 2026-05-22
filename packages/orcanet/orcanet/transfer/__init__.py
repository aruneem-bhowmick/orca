"""Transfer strategies for cross-domain knowledge transfer."""

from orcanet.transfer.architecture_transfer import ArchitectureTransfer, adapt_architecture
from orcanet.transfer.base import TransferStrategy
from orcanet.transfer.feature_transfer import FeatureTransfer, linear_cka
from orcanet.transfer.multi_task_transfer import MultiTaskModel, MultiTaskTransfer
from orcanet.transfer.types import TransferScore
from orcanet.transfer.weight_transfer import WeightTransfer, get_optimizer_with_layer_lr

__all__ = [
    "ArchitectureTransfer",
    "FeatureTransfer",
    "MultiTaskModel",
    "MultiTaskTransfer",
    "TransferScore",
    "TransferStrategy",
    "WeightTransfer",
    "adapt_architecture",
    "get_optimizer_with_layer_lr",
    "linear_cka",
]
