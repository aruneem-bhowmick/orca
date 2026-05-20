"""Domain-Adversarial Neural Network (DANN) cross-domain task embedder.

Reference: Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class GradientReversalFunction(torch.autograd.Function):
    """Custom autograd primitive: identity in the forward pass, gradient negation in backward."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, alpha: float) -> Tensor:  # type: ignore[override]
        """Store *alpha* for backward and return *x* unchanged."""
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:  # type: ignore[override]
        """Return ``-alpha * grad_output`` so downstream gradients are negated and scaled."""
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """Passes input unchanged in the forward pass; negates and scales gradients in backward."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialise with gradient reversal scale factor *alpha* (default 1.0)."""
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        """Apply gradient reversal: identity forward, ``-alpha``-scaled backward."""
        return GradientReversalFunction.apply(x, self.alpha)


class _FeatureMLP(nn.Module):
    """Feed-forward MLP: input_dim → hidden_dims → output_dim, without output normalisation."""

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int) -> None:
        """Build layer stack: ``Linear → BatchNorm1d → ReLU`` per hidden dim, then final Linear."""
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Run *x* through the MLP and return unnormalised output features."""
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
        """Construct the shared feature extractor, task classifier, and adversarial domain head."""
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
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                features = self.feature_extractor(x)
        finally:
            if was_training:
                self.train()
        return F.normalize(features, p=2, dim=1)

    def fit(
        self,
        x: Tensor,
        task_labels: Tensor,
        domain_labels: Tensor,
        epochs: int = 20,
        lr: float = 1e-3,
        domain_lambda: float = 1.0,
    ) -> dict[str, list[float]]:
        """Train with the DANN objective using progressive GRL alpha scheduling.

        Alpha follows the Ganin et al. schedule: α(p) = 2/(1+exp(−10p))−1 where
        p goes from 0 to 1 over training, so the domain adversary strengthens gradually.

        Args:
            x:              Input tensor of shape (N, input_dim).
            task_labels:    Long tensor of shape (N,) with task-type class indices.
            domain_labels:  Long tensor of shape (N,) with domain indices.
            epochs:         Number of training epochs.
            lr:             Adam learning rate.
            domain_lambda:  Weight λ for the domain adversarial loss term.

        Returns:
            Dict with keys ``"task_loss"`` and ``"domain_loss"``, each a list of
            per-epoch scalar losses.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        task_loss_history: list[float] = []
        domain_loss_history: list[float] = []

        for epoch in range(1, epochs + 1):
            self.train()

            p = (epoch - 1) / max(epochs - 1, 1)
            self._grl.alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

            optimizer.zero_grad()
            features, task_logits, domain_logits = self(x)

            l_task = F.cross_entropy(task_logits, task_labels)
            l_domain = F.cross_entropy(domain_logits, domain_labels)
            (l_task + domain_lambda * l_domain).backward()
            optimizer.step()

            task_loss_history.append(float(l_task))
            domain_loss_history.append(float(l_domain))

            logger.debug(
                "epoch %d/%d  alpha=%.4f  task_loss=%.4f  domain_loss=%.4f",
                epoch, epochs, self._grl.alpha, float(l_task), float(l_domain),
            )

        return {"task_loss": task_loss_history, "domain_loss": domain_loss_history}

    def save(self, path: str | Path) -> None:
        """Serialise model weights and constructor config to *path*."""
        torch.save(
            {
                "config": {
                    "input_dim": self._input_dim,
                    "embedding_dim": self._embedding_dim,
                    "n_domains": self._n_domains,
                    "n_task_types": self._n_task_types,
                    "hidden_dims": self._hidden_dims,
                },
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> CrossDomainEmbedder:
        """Load a saved model from *path* and return it in eval mode."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(**checkpoint["config"])
        obj.load_state_dict(checkpoint["state_dict"])
        obj.eval()
        return obj
