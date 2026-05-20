"""Domain-Adversarial Neural Network (DANN) cross-domain task embedder.

Reference: Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, alpha: float) -> Tensor:  # type: ignore[override]
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """Passes input unchanged in the forward pass; negates and scales gradients in backward."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


class _FeatureMLP(nn.Module):
    """Feed-forward MLP: input_dim → hidden_dims → output_dim, without output normalisation."""

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CrossDomainEmbedder(nn.Module):
    """DANN-based embedder that learns domain-invariant 64-dim task representations.

    The gradient reversal layer (GRL) between the shared feature extractor and the
    domain classifier forces the extractor to produce representations that are useful
    for task-type classification but indistinguishable across domains.

    Args:
        input_dim:      Dimensionality of input meta-features (25 from StatisticalEmbedder).
        embedding_dim:  Output embedding dimensionality.
        n_domains:      Number of source domains for the domain classifier head.
        n_task_types:   Number of task-type classes for the task classifier head.
        hidden_dims:    Hidden layer sizes for the shared feature extractor.
    """

    def __init__(
        self,
        input_dim: int = 25,
        embedding_dim: int = 64,
        n_domains: int = 10,
        n_task_types: int = 3,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self._input_dim = input_dim
        self._embedding_dim = embedding_dim
        self._n_domains = n_domains
        self._n_task_types = n_task_types
        self._hidden_dims = list(hidden_dims)

        self.feature_extractor = _FeatureMLP(input_dim, hidden_dims, embedding_dim)
        self.task_classifier = nn.Linear(embedding_dim, n_task_types)

        # Store as named attribute so alpha can be updated during training;
        # the Sequential holds the same object reference, so updates propagate.
        self._grl = GradientReversalLayer(alpha=1.0)
        self.domain_classifier = nn.Sequential(
            self._grl,
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_domains),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return (features, task_logits, domain_logits)."""
        features = self.feature_extractor(x)
        task_logits = self.task_classifier(features)
        domain_logits = self.domain_classifier(features)
        return features, task_logits, domain_logits

    def embed(self, x: Tensor) -> Tensor:
        """Return L2-normalised feature embeddings (eval mode, no gradient tracking)."""
        self.eval()
        with torch.no_grad():
            features = self.feature_extractor(x)
        return F.normalize(features, p=2, dim=1)
