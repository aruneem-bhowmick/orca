"""Unit tests for CrossDomainEmbedder and GradientReversalLayer."""

from __future__ import annotations

import torch
import pytest

from orcanet.embeddings.cross_domain import CrossDomainEmbedder, GradientReversalLayer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_batch(
    n_samples: int = 60,
    n_domains: int = 5,
    n_task_types: int = 3,
    input_dim: int = 25,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn(n_samples, input_dim)
    task_labels = torch.randint(0, n_task_types, (n_samples,))
    domain_labels = torch.randint(0, n_domains, (n_samples,))
    return x, task_labels, domain_labels


# ---------------------------------------------------------------------------
# Test 1: Output shape and L2 normalisation
# ---------------------------------------------------------------------------


class TestShapeAndNormalisation:
    def test_embed_output_shape(self) -> None:
        model = CrossDomainEmbedder(input_dim=25, embedding_dim=64)
        x = torch.randn(16, 25)
        out = model.embed(x)
        assert out.shape == (16, 64)

    def test_embed_l2_normalised(self) -> None:
        model = CrossDomainEmbedder(input_dim=25, embedding_dim=64)
        x = torch.randn(32, 25)
        out = model.embed(x)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(32), atol=1e-5)

    def test_forward_output_shapes(self) -> None:
        model = CrossDomainEmbedder(
            input_dim=25, embedding_dim=64, n_domains=10, n_task_types=3
        )
        x = torch.randn(8, 25)
        features, task_logits, domain_logits = model(x)
        assert features.shape == (8, 64)
        assert task_logits.shape == (8, 3)
        assert domain_logits.shape == (8, 10)

    def test_embed_no_gradient(self) -> None:
        model = CrossDomainEmbedder()
        x = torch.randn(4, 25)
        out = model.embed(x)
        assert not out.requires_grad


# ---------------------------------------------------------------------------
# Test 2: GRL gradient negation (using register_hook per spec)
# ---------------------------------------------------------------------------


class TestGradientReversal:
    def test_grl_negates_gradient(self) -> None:
        grl = GradientReversalLayer(alpha=1.0)
        x = torch.ones(4, 8, requires_grad=True)
        captured: list[torch.Tensor] = []
        x.register_hook(lambda g: captured.append(g.clone()))

        grl(x).sum().backward()

        assert len(captured) == 1
        assert torch.allclose(captured[0], -torch.ones_like(x))

    def test_grl_scales_by_alpha(self) -> None:
        alpha = 2.5
        grl = GradientReversalLayer(alpha=alpha)
        x = torch.ones(3, 5, requires_grad=True)
        captured: list[torch.Tensor] = []
        x.register_hook(lambda g: captured.append(g.clone()))

        grl(x).sum().backward()

        assert torch.allclose(captured[0], -alpha * torch.ones_like(x))

    def test_grl_forward_is_identity(self) -> None:
        grl = GradientReversalLayer(alpha=0.5)
        x = torch.randn(6, 12)
        assert torch.allclose(grl(x), x)

    def test_grl_alpha_zero_passes_gradient_unchanged(self) -> None:
        grl = GradientReversalLayer(alpha=0.0)
        x = torch.ones(2, 4, requires_grad=True)
        captured: list[torch.Tensor] = []
        x.register_hook(lambda g: captured.append(g.clone()))

        grl(x).sum().backward()

        assert torch.allclose(captured[0], torch.zeros_like(x))
