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

    def test_embed_preserves_training_mode(self) -> None:
        model = CrossDomainEmbedder()
        model.train()
        model.embed(torch.randn(4, 25))
        assert model.training, "embed() must restore training mode after execution"

    def test_embed_preserves_eval_mode(self) -> None:
        model = CrossDomainEmbedder()
        model.eval()
        model.embed(torch.randn(4, 25))
        assert not model.training, "embed() must not flip a model that was already in eval mode"


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


# ---------------------------------------------------------------------------
# Test 3: Training convergence
# ---------------------------------------------------------------------------


class TestTrainingConvergence:
    def test_task_loss_decreases_over_epochs(self) -> None:
        torch.manual_seed(42)
        model = CrossDomainEmbedder(
            input_dim=25, embedding_dim=64, n_domains=5, n_task_types=3
        )
        x, task_labels, domain_labels = _make_batch(n_samples=60, n_domains=5, n_task_types=3)
        history = model.fit(x, task_labels, domain_labels, epochs=20, lr=1e-3)

        task_losses = history["task_loss"]
        assert len(task_losses) == 20
        assert task_losses[-1] < task_losses[0], (
            f"Task loss did not decrease: first={task_losses[0]:.4f}, last={task_losses[-1]:.4f}"
        )

    def test_fit_returns_correct_structure(self) -> None:
        model = CrossDomainEmbedder()
        x, task_labels, domain_labels = _make_batch()
        history = model.fit(x, task_labels, domain_labels, epochs=5)

        assert set(history.keys()) == {"task_loss", "domain_loss"}
        assert len(history["task_loss"]) == 5
        assert len(history["domain_loss"]) == 5
        assert all(isinstance(v, float) for v in history["task_loss"])


# ---------------------------------------------------------------------------
# Test 4: Domain invariance
# ---------------------------------------------------------------------------


class TestDomainInvariance:
    def test_within_vs_cross_domain_spread(self) -> None:
        """After training, within-domain and cross-domain cosine distance spreads should
        be of similar magnitude (ratio between 0.3 and 3.0), demonstrating that the
        embedder is not trivially clustering by domain identity.
        """
        torch.manual_seed(7)
        n_domains = 4
        samples_per_domain = 15
        input_dim = 25

        # Domain-shifted data: each domain has a distinct mean offset
        x_parts = []
        domain_labels_list: list[int] = []
        task_labels_list: list[int] = []
        for d in range(n_domains):
            domain_x = torch.randn(samples_per_domain, input_dim) + d * 0.5
            x_parts.append(domain_x)
            domain_labels_list.extend([d] * samples_per_domain)
            task_labels_list.extend([d % 3] * samples_per_domain)

        x = torch.cat(x_parts, dim=0)
        domain_labels = torch.tensor(domain_labels_list)
        task_labels = torch.tensor(task_labels_list)

        model = CrossDomainEmbedder(
            input_dim=input_dim, embedding_dim=64, n_domains=n_domains, n_task_types=3
        )
        model.fit(x, task_labels, domain_labels, epochs=20, lr=1e-3)

        embeddings = model.embed(x)  # (N, 64), L2-normalised
        n_samples = embeddings.shape[0]
        sim_matrix = embeddings @ embeddings.T  # cosine similarities

        within_dists: list[float] = []
        cross_dists: list[float] = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = float(1.0 - sim_matrix[i, j].item())
                if domain_labels[i] == domain_labels[j]:
                    within_dists.append(dist)
                else:
                    cross_dists.append(dist)

        within_std = float(torch.tensor(within_dists).std())
        cross_std = float(torch.tensor(cross_dists).std())
        ratio = within_std / (cross_std + 1e-8)

        assert 0.3 <= ratio <= 3.0, (
            f"Domain invariance check failed: within_std={within_std:.4f}, "
            f"cross_std={cross_std:.4f}, ratio={ratio:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 5: Save / load persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path: pytest.TempPathFactory) -> None:
        torch.manual_seed(0)
        model = CrossDomainEmbedder(input_dim=25, embedding_dim=64, n_domains=5, n_task_types=3)
        x, task_labels, domain_labels = _make_batch(n_samples=30, n_domains=5, n_task_types=3)
        model.fit(x, task_labels, domain_labels, epochs=3)

        path = tmp_path / "embedder.pt"  # type: ignore[operator]
        model.save(path)

        loaded = CrossDomainEmbedder.load(path)
        x_test = torch.randn(8, 25)
        original_out = model.embed(x_test)
        loaded_out = loaded.embed(x_test)
        assert torch.allclose(original_out, loaded_out, atol=1e-5), (
            "Loaded model embeddings do not match saved model embeddings"
        )

    def test_load_preserves_config(self, tmp_path: pytest.TempPathFactory) -> None:
        model = CrossDomainEmbedder(
            input_dim=20, embedding_dim=32, n_domains=6, n_task_types=4, hidden_dims=[64, 32]
        )
        path = tmp_path / "embedder_cfg.pt"  # type: ignore[operator]
        model.save(path)

        loaded = CrossDomainEmbedder.load(path)
        assert loaded._input_dim == 20
        assert loaded._embedding_dim == 32
        assert loaded._n_domains == 6
        assert loaded._n_task_types == 4
        assert loaded._hidden_dims == [64, 32]
