"""Unit tests for NeuralEmbedder and ContrastiveLoss."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from orcamind.embedders.neural import ContrastiveLoss, NeuralEmbedder

_OUTPUT_DIM = 64
_N_DOMAINS = 3
_DS_PER_DOMAIN = 4


def _make_multi_domain_datasets(
    n_domains: int = _N_DOMAINS,
    ds_per_domain: int = _DS_PER_DOMAIN,
    n_rows: int = 50,
    n_cols: int = 5,
    seed: int = 0,
) -> tuple[list[tuple[pd.DataFrame, pd.Series]], list[int]]:
    rng = np.random.default_rng(seed)
    dataset_list: list[tuple[pd.DataFrame, pd.Series]] = []
    domain_labels: list[int] = []
    for domain in range(n_domains):
        for _ in range(ds_per_domain):
            X = pd.DataFrame(
                rng.standard_normal((n_rows, n_cols)),
                columns=[f"f{i}" for i in range(n_cols)],
            )
            y = pd.Series(rng.integers(0, 2, size=n_rows), name="target")
            dataset_list.append((X, y))
            domain_labels.append(domain)
    return dataset_list, domain_labels


@pytest.fixture()
def embedder() -> NeuralEmbedder:
    return NeuralEmbedder()


@pytest.fixture()
def sample_dataset() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((60, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(rng.integers(0, 3, size=60), name="target")
    return X, y


class TestOutputShape:
    def test_embed_single_shape(self, embedder: NeuralEmbedder, sample_dataset: tuple) -> None:
        X, y = sample_dataset
        vec = embedder.embed(X, y)
        assert vec.shape == (_OUTPUT_DIM,)

    def test_embed_batch_shape(self, embedder: NeuralEmbedder, sample_dataset: tuple) -> None:
        X, y = sample_dataset
        result = embedder.embed_batch([(X, y)] * 5)
        assert result.shape == (5, _OUTPUT_DIM)

    def test_embedding_dim_property(self, embedder: NeuralEmbedder) -> None:
        assert embedder.embedding_dim == _OUTPUT_DIM

    def test_embed_batch_empty(self, embedder: NeuralEmbedder) -> None:
        result = embedder.embed_batch([])
        assert result.shape == (0, _OUTPUT_DIM)

    def test_embed_no_labels(self, embedder: NeuralEmbedder, sample_dataset: tuple) -> None:
        X, _ = sample_dataset
        vec = embedder.embed(X, None)
        assert vec.shape == (_OUTPUT_DIM,)


class TestL2Normalization:
    def test_embed_unit_norm(self, embedder: NeuralEmbedder, sample_dataset: tuple) -> None:
        X, y = sample_dataset
        vec = embedder.embed(X, y)
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5

    def test_embed_batch_unit_norms(self, embedder: NeuralEmbedder, sample_dataset: tuple) -> None:
        X, y = sample_dataset
        result = embedder.embed_batch([(X, y)] * 8)
        norms = np.linalg.norm(result, axis=1)
        assert np.all(np.abs(norms - 1.0) < 1e-5)


class TestContrastiveLoss:
    def test_loss_decreases_over_training(self) -> None:
        torch.manual_seed(0)
        embedder = NeuralEmbedder()
        dataset_list, domain_labels = _make_multi_domain_datasets()

        stat = embedder._stat_embedder.embed_batch(dataset_list)
        X = torch.tensor(stat, dtype=torch.float32)
        y = torch.tensor(domain_labels, dtype=torch.long)

        loss_fn = ContrastiveLoss()
        optimizer = torch.optim.Adam(embedder._mlp.parameters(), lr=1e-3)
        embedder._mlp.train()

        losses: list[float] = []
        for _ in range(10):
            optimizer.zero_grad()
            z = embedder._mlp(X)
            loss = loss_fn(z, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        assert losses[-1] < losses[0], f"loss did not decrease: {losses}"

    def test_loss_zero_when_no_positive_pairs(self) -> None:
        embeddings = torch.randn(4, 64)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        labels = torch.tensor([0, 1, 2, 3])  # all distinct
        loss_fn = ContrastiveLoss()
        loss = loss_fn(embeddings, labels)
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-6)

    def test_loss_positive_with_valid_pairs(self) -> None:
        torch.manual_seed(1)
        embeddings = torch.nn.functional.normalize(torch.randn(6, 64), p=2, dim=1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        loss_fn = ContrastiveLoss()
        loss = loss_fn(embeddings, labels)
        assert loss.item() > 0.0

    def test_loss_single_sample_returns_zero(self) -> None:
        embeddings = torch.nn.functional.normalize(torch.randn(1, 64), p=2, dim=1)
        labels = torch.tensor([0])
        loss_fn = ContrastiveLoss()
        loss = loss_fn(embeddings, labels)
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-6)


class TestPersistence:
    def test_save_load_roundtrip(
        self, tmp_path: object, embedder: NeuralEmbedder, sample_dataset: tuple
    ) -> None:
        from pathlib import Path

        dataset_list, domain_labels = _make_multi_domain_datasets()
        embedder.fit(dataset_list, domain_labels, epochs=3, lr=1e-3)

        save_path = str(Path(str(tmp_path)) / "model.pt")
        embedder.save(save_path)

        loaded = NeuralEmbedder.load(save_path)
        X, y = sample_dataset
        out_original = embedder.embed(X, y)
        out_loaded = loaded.embed(X, y)
        assert np.allclose(out_original, out_loaded, atol=1e-5)

    def test_load_preserves_config(self, tmp_path: object) -> None:
        from pathlib import Path

        original = NeuralEmbedder(input_dim=25, hidden_dims=[64, 32], output_dim=32)
        save_path = str(Path(str(tmp_path)) / "model_cfg.pt")
        original.save(save_path)

        loaded = NeuralEmbedder.load(save_path)
        assert loaded.embedding_dim == 32
        assert loaded._hidden_dims == [64, 32]
        assert loaded._input_dim == 25


class TestFitValidation:
    def test_fit_raises_on_length_mismatch(self, embedder: NeuralEmbedder) -> None:
        dataset_list, domain_labels = _make_multi_domain_datasets()
        with pytest.raises(ValueError, match="length"):
            embedder.fit(dataset_list, domain_labels[:-1], epochs=1)

    def test_fit_raises_on_single_dataset(
        self, embedder: NeuralEmbedder, sample_dataset: tuple
    ) -> None:
        X, y = sample_dataset
        with pytest.raises(ValueError, match="at least 2"):
            embedder.fit([(X, y)], [0], epochs=1)
