"""Fixtures for orcanet embeddings unit tests."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


class _DeterministicSentenceTransformer:
    """Offline-capable SentenceTransformer stub.

    Produces 384-dim vectors from a small keyword vocabulary so that semantic
    similarity tests remain valid without a network connection.
    """

    _KEYWORDS: tuple[str, ...] = (
        # vision / image tasks (indices 0-5)
        "image", "visual", "classification", "recognition", "photo", "picture",
        # financial / time-series tasks (indices 6-11)
        "financial", "time", "series", "regression", "forecast", "stock",
        # medical tasks (indices 12-17)
        "medical", "clinical", "health", "brain", "mri", "scan",
        # NLP tasks (indices 18-23)
        "text", "nlp", "language", "sentence", "word", "document",
        # general ML qualifiers (indices 24-35)
        "binary", "multi", "class", "label", "category", "predict",
        "task", "model", "neural", "deep", "learning", "train",
    )

    def __init__(self, model_name: str | None = None, **kwargs: object) -> None:
        """Accept and ignore the model_name and any extra SentenceTransformer kwargs."""
        self._dim = 384

    def encode(
        self,
        sentences: str | list[str],
        *,
        normalize_embeddings: bool = False,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        **kwargs: object,
    ) -> np.ndarray:
        """Return keyword-based embeddings for *sentences* (string or list of strings)."""
        if isinstance(sentences, str):
            return self._encode_one(sentences)
        return np.stack([self._encode_one(s) for s in sentences])

    def _encode_one(self, text: str) -> np.ndarray:
        """Return a deterministic 384-dim L2-normalised vector for *text*."""
        vec = np.zeros(self._dim, dtype=np.float32)
        words = text.lower().split()
        for idx, kw in enumerate(self._KEYWORDS):
            if any(kw in w for w in words):
                vec[idx] = 1.0
        # Deterministic low-amplitude noise for non-keyword dimensions.
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        noise_len = self._dim - len(self._KEYWORDS)
        vec[len(self._KEYWORDS):] = rng.randn(noise_len).astype(np.float32) * 0.05
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec


@pytest.fixture(scope="session", autouse=True)
def _patch_sentence_transformers() -> object:
    """Patch SentenceTransformer globally so all embeddings tests run offline without model weights."""
    with patch(
        "orcanet.embeddings.text_features.SentenceTransformer",
        _DeterministicSentenceTransformer,
    ):
        yield
