"""Local transfer-scoring types for the OrcaNet transfer module."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TransferScore:
    """Layer-level CKA transfer score between a source and target task.

    This is distinct from ``orca_shared.schemas.transfer.TransferScore``, which
    is a lightweight API schema.  This richer dataclass carries per-layer detail
    used internally by transfer strategies.
    """

    overall: float
    layer_scores: dict[str, float] = field(default_factory=dict)
    recommended_layers: list[str] = field(default_factory=list)
    reasoning: str = ""
