"""Cross-domain embedding quality benchmark for CrossDomainEmbedder.

Verifies domain-invariance properties of the DANN-based embedder:

1. After training on 40 tasks from 5 domains, the absolute difference between
   within-domain mean cosine similarity and cross-domain mean cosine similarity
   on 10 held-out tasks is < 0.15  (domain invariance).

2. ``CrossDomainEmbedder`` (DANN adversarial) produces a smaller domain gap
   than ``NeuralEmbedder`` (NT-Xent contrastive), which is designed to cluster
   same-domain tasks together.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
import torch

from orcamind.embedders.neural import NeuralEmbedder
from orcamind.embedders.statistical import StatisticalEmbedder
from orcanet.embeddings.cross_domain import CrossDomainEmbedder

pytestmark = pytest.mark.benchmark

# ---------------------------------------------------------------------------
# Benchmark hyper-parameters
# ---------------------------------------------------------------------------

_N_DOMAINS: int = 5
_N_PER_DOMAIN: int = 10  # tasks per domain
_N_TRAIN_PER_DOMAIN: int = 8  # tasks used for training
_N_TEST_PER_DOMAIN: int = _N_PER_DOMAIN - _N_TRAIN_PER_DOMAIN  # held-out

_DANN_EPOCHS: int = 80
_NEURAL_EPOCHS: int = 80
_DOMAIN_GAP_THRESHOLD: float = 0.15  # CrossDomainEmbedder must stay below this


# ---------------------------------------------------------------------------
# Synthetic dataset factory
# ---------------------------------------------------------------------------


def _make_domain_dataset(domain_id: int, task_idx: int) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic tabular dataset with domain-specific statistical properties.

    Each domain is designed to produce a clearly distinct StatisticalEmbedder
    meta-feature vector, ensuring that training signals are strong.

    Args:
        domain_id: Integer index of the domain (0–4).
        task_idx:  Within-domain task index (0–9); used as a random-seed offset
                   to ensure tasks within the same domain are similar but not
                   identical.

    Returns:
        A ``(DataFrame, Series)`` pair suitable for ``StatisticalEmbedder.embed``.
    """
    rng = np.random.default_rng(domain_id * 100 + task_idx)

    if domain_id == 0:
        # Medical-like: few features, normal distribution, balanced binary classes.
        n, d = 300 + rng.integers(-20, 20), 8
        X = rng.normal(0.0, 1.0, (n, d))
        y = rng.integers(0, 2, n)

    elif domain_id == 1:
        # Finance-like: many correlated features, skewed values, binary outcome.
        n, d = 200 + rng.integers(-15, 15), 20
        cov = 0.85 * np.ones((d, d)) + 0.15 * np.eye(d)
        X = rng.multivariate_normal(np.zeros(d), cov, n)
        X = np.abs(X) * rng.exponential(1.0, (n, d))  # skewed positive values
        y = rng.integers(0, 2, n)

    elif domain_id == 2:
        # Vision-like: high-dimensional, very sparse (70 % zeros), many classes.
        n, d = 400 + rng.integers(-30, 30), 64
        X = rng.uniform(0.0, 1.0, (n, d))
        X[X < 0.7] = 0.0  # sparsify
        y = rng.integers(0, 8, n)

    elif domain_id == 3:
        # NLP-like: strongly bimodal clusters, clear class separability.
        n, d = 250 + rng.integers(-20, 20), 16
        half = n // 2
        X = np.vstack(
            [rng.normal(-4.0, 0.6, (half, d)), rng.normal(4.0, 0.6, (n - half, d))]
        )
        y = np.concatenate([np.zeros(half, dtype=int), np.ones(n - half, dtype=int)])

    else:
        # Tabular-like: heavy-tailed (t-distributed), diverse scale mixing, multi-class.
        n, d = 350 + rng.integers(-25, 25), 24
        X = rng.standard_t(df=2.0, size=(n, d))
        X[:, : d // 4] *= 100.0  # mix feature scales
        y = rng.integers(0, 6, n)

    df = pd.DataFrame(X.astype(np.float64), columns=[f"f{i}" for i in range(X.shape[1])])
    labels = pd.Series(y.astype(int), name="label")
    return df, labels


# ---------------------------------------------------------------------------
# Cosine-similarity helpers
# ---------------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two 1-D arrays."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def _domain_gap(
    embeddings: np.ndarray,
    domain_ids: list[int],
) -> float:
    """Compute the absolute domain gap on held-out embeddings.

    The domain gap is ``|mean_within_domain_cosine - mean_cross_domain_cosine|``.
    A value close to 0 indicates domain invariance.

    Args:
        embeddings:  Array of shape ``(N, embedding_dim)`` — one row per task.
        domain_ids:  List of length N mapping each row to a domain index.

    Returns:
        Non-negative float: the absolute gap between within- and cross-domain
        mean cosine similarities.
    """
    n = len(domain_ids)
    within_scores: list[float] = []
    cross_scores: list[float] = []

    for i, j in combinations(range(n), 2):
        score = _cosine(embeddings[i], embeddings[j])
        if domain_ids[i] == domain_ids[j]:
            within_scores.append(score)
        else:
            cross_scores.append(score)

    within_mean = float(np.mean(within_scores)) if within_scores else 0.0
    cross_mean = float(np.mean(cross_scores)) if cross_scores else 0.0
    return abs(within_mean - cross_mean)


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


class TestCrossDomainEmbeddingQuality:
    """Validates domain-invariance properties of the CrossDomainEmbedder."""

    @pytest.fixture(scope="class")
    def datasets(self) -> list[tuple[tuple[pd.DataFrame, pd.Series], int, int]]:
        """Generate all 50 (dataset, domain_id, task_idx) triples once per class."""
        items = []
        for domain_id in range(_N_DOMAINS):
            for task_idx in range(_N_PER_DOMAIN):
                ds = _make_domain_dataset(domain_id, task_idx)
                items.append((ds, domain_id, task_idx))
        return items

    @pytest.fixture(scope="class")
    def stat_embedder(self) -> StatisticalEmbedder:
        """Provide a single shared StatisticalEmbedder instance."""
        return StatisticalEmbedder()

    @pytest.fixture(scope="class")
    def feature_vectors(
        self,
        datasets: list[tuple[tuple[pd.DataFrame, pd.Series], int, int]],
        stat_embedder: StatisticalEmbedder,
    ) -> np.ndarray:
        """Pre-compute all 50 × 25 statistical meta-feature vectors."""
        vecs = [stat_embedder.embed(df, labels) for (df, labels), _, _ in datasets]
        return np.stack(vecs, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # CrossDomainEmbedder test
    # ------------------------------------------------------------------

    def test_domain_gap_below_threshold(
        self,
        datasets: list[tuple[tuple[pd.DataFrame, pd.Series], int, int]],
        feature_vectors: np.ndarray,
    ) -> None:
        """CrossDomainEmbedder domain gap must be < 0.15 after DANN training.

        Train on 40 tasks (8 per domain) and evaluate on 10 held-out (2 per domain).
        """
        domain_ids = [d for _, d, _ in datasets]

        # Split into train/test
        train_idx = [
            i for i in range(len(datasets))
            if datasets[i][2] < _N_TRAIN_PER_DOMAIN  # task_idx < 8
        ]
        test_idx = [i for i in range(len(datasets)) if i not in set(train_idx)]

        x_train = torch.tensor(feature_vectors[train_idx], dtype=torch.float32)
        task_labels_train = torch.zeros(len(train_idx), dtype=torch.long)
        domain_labels_train = torch.tensor(
            [domain_ids[i] for i in train_idx], dtype=torch.long
        )

        torch.manual_seed(7)
        embedder = CrossDomainEmbedder(
            input_dim=25,
            embedding_dim=64,
            n_domains=_N_DOMAINS,
            n_task_types=2,
        )
        embedder.fit(
            x_train,
            task_labels_train,
            domain_labels_train,
            epochs=_DANN_EPOCHS,
            lr=5e-3,
        )

        x_test = torch.tensor(feature_vectors[test_idx], dtype=torch.float32)
        test_embs = embedder.embed(x_test).numpy()
        test_domain_ids = [domain_ids[i] for i in test_idx]

        gap = _domain_gap(test_embs, test_domain_ids)
        assert gap < _DOMAIN_GAP_THRESHOLD, (
            f"CrossDomainEmbedder domain gap = {gap:.4f} exceeds threshold "
            f"{_DOMAIN_GAP_THRESHOLD}"
        )

    # ------------------------------------------------------------------
    # CrossDomainEmbedder vs NeuralEmbedder comparison
    # ------------------------------------------------------------------

    def test_dann_gap_smaller_than_contrastive_gap(
        self,
        datasets: list[tuple[tuple[pd.DataFrame, pd.Series], int, int]],
        feature_vectors: np.ndarray,
    ) -> None:
        """CrossDomainEmbedder (adversarial) must produce a smaller domain gap than
        NeuralEmbedder (contrastive).

        The DANN adversarial objective removes domain information from the
        feature extractor, while the NT-Xent contrastive loss groups same-domain
        tasks together — creating a larger domain gap.
        """
        domain_ids = [d for _, d, _ in datasets]

        train_idx = [i for i in range(len(datasets)) if datasets[i][2] < _N_TRAIN_PER_DOMAIN]
        test_idx = [i for i in range(len(datasets)) if i not in set(train_idx)]

        train_domain_ids = [domain_ids[i] for i in train_idx]
        test_domain_ids = [domain_ids[i] for i in test_idx]

        # ---- CrossDomainEmbedder ----
        x_train = torch.tensor(feature_vectors[train_idx], dtype=torch.float32)
        task_labels_train = torch.zeros(len(train_idx), dtype=torch.long)
        domain_labels_train = torch.tensor(train_domain_ids, dtype=torch.long)

        torch.manual_seed(7)
        dann = CrossDomainEmbedder(
            input_dim=25, embedding_dim=64, n_domains=_N_DOMAINS, n_task_types=2
        )
        dann.fit(
            x_train,
            task_labels_train,
            domain_labels_train,
            epochs=_DANN_EPOCHS,
            lr=5e-3,
        )
        x_test = torch.tensor(feature_vectors[test_idx], dtype=torch.float32)
        dann_embs = dann.embed(x_test).numpy()
        dann_gap = _domain_gap(dann_embs, test_domain_ids)

        # ---- NeuralEmbedder (contrastive) ----
        train_datasets = [datasets[i][0] for i in train_idx]
        torch.manual_seed(7)
        neural = NeuralEmbedder(input_dim=25, hidden_dims=[128, 64], output_dim=64)
        neural.fit(
            dataset_list=train_datasets,
            domain_labels=train_domain_ids,
            epochs=_NEURAL_EPOCHS,
            lr=5e-3,
        )
        test_datasets = [datasets[i][0] for i in test_idx]
        neural_embs = np.stack(
            [neural.embed(df, labels) for df, labels in test_datasets], axis=0
        )
        neural_gap = _domain_gap(neural_embs, test_domain_ids)

        assert dann_gap <= neural_gap, (
            f"Expected CrossDomainEmbedder gap ({dann_gap:.4f}) ≤ "
            f"NeuralEmbedder gap ({neural_gap:.4f}), but it was larger. "
            "DANN adversarial training should produce more domain-invariant embeddings."
        )

    def test_embeddings_are_l2_normalised(
        self,
        feature_vectors: np.ndarray,
    ) -> None:
        """CrossDomainEmbedder.embed() must return L2-normalised vectors."""
        embedder = CrossDomainEmbedder(input_dim=25)
        x = torch.tensor(feature_vectors[:5], dtype=torch.float32)
        embs = embedder.embed(x).numpy()
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, np.ones(5), atol=1e-5)

    def test_embedding_shape_is_correct(
        self,
        feature_vectors: np.ndarray,
    ) -> None:
        """CrossDomainEmbedder.embed() must return shape (N, 64)."""
        embedder = CrossDomainEmbedder(input_dim=25, embedding_dim=64)
        x = torch.tensor(feature_vectors[:10], dtype=torch.float32)
        embs = embedder.embed(x)
        assert embs.shape == (10, 64)
