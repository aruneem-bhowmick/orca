"""Text-based task description embedder with optional statistical feature fusion."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from orcamind.embedders.statistical import StatisticalEmbedder

logger = logging.getLogger(__name__)

_TEXT_DIM = 384
_STAT_DIM = 25


class _AddFusion(nn.Module):
    """Projects text and stat vectors to output_dim independently, then sums them."""

    def __init__(self, text_dim: int, stat_dim: int, output_dim: int) -> None:
        """Initialise two independent linear projections for text and stat inputs."""
        super().__init__()
        self.text_proj = nn.Linear(text_dim, output_dim)
        self.stat_proj = nn.Linear(stat_dim, output_dim)

    def forward(self, x_text: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        """Return the element-wise sum of the text and stat projections."""
        return self.text_proj(x_text) + self.stat_proj(x_stat)


class _AttentionFusion(nn.Module):
    """Softmax-gated weighted sum of text and stat projections."""

    def __init__(self, text_dim: int, stat_dim: int, output_dim: int) -> None:
        """Initialise text/stat projections and a 2-head attention gate."""
        super().__init__()
        self.text_proj = nn.Linear(text_dim, output_dim)
        self.stat_proj = nn.Linear(stat_dim, output_dim)
        self.attn = nn.Linear(2 * output_dim, 2)

    def forward(self, x_text: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        """Return a softmax-gated weighted combination of the text and stat projections."""
        t = self.text_proj(x_text)
        s = self.stat_proj(x_stat)
        weights = F.softmax(self.attn(torch.cat([t, s], dim=-1)), dim=-1)
        return weights[..., 0:1] * t + weights[..., 1:2] * s


class TextTaskEmbedder:
    """Embeds natural language task descriptions and optionally fuses with statistical features.

    Args:
        model_name:            SentenceTransformer model identifier.
        statistical_embedder:  Pre-built StatisticalEmbedder; instantiated fresh if None.
        fusion:                How to combine text (384-dim) and stat (25-dim) vectors.
                               One of ``"concat"``, ``"add"``, or ``"attention"``.
        output_dim:            Dimensionality of the fused output from ``embed_with_stats``.
    """

    _VALID_FUSIONS = frozenset({"concat", "add", "attention"})

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        statistical_embedder: StatisticalEmbedder | None = None,
        fusion: str = "concat",
        output_dim: int = 128,
    ) -> None:
        """Load the SentenceTransformer model and build the configured fusion network.

        Raises:
            ValueError: If *fusion* is not one of ``"concat"``, ``"add"``, or ``"attention"``.
        """
        if fusion not in self._VALID_FUSIONS:
            raise ValueError(
                f"fusion must be one of {sorted(self._VALID_FUSIONS)!r}, got {fusion!r}"
            )

        self.st_model = SentenceTransformer(model_name)
        self._statistical_embedder = statistical_embedder or StatisticalEmbedder()
        self._fusion = fusion
        self._output_dim = output_dim
        self._fusion_net: nn.Module = self._build_fusion(fusion, output_dim)
        self._fusion_net.eval()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed_from_description(self, description: str) -> np.ndarray:
        """Encode *description* into a 384-dim L2-normalised semantic vector."""
        vec: np.ndarray = self.st_model.encode(description, convert_to_numpy=True)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_with_stats(
        self,
        description: str,
        dataset: pd.DataFrame,
        labels: pd.Series | None = None,
    ) -> np.ndarray:
        """Fuse text description embedding with dataset statistical features.

        Returns an L2-normalised vector of shape ``(output_dim,)``.
        """
        text_vec = self.embed_from_description(description)
        stat_vec = self._statistical_embedder.embed(dataset, labels)

        x_text = torch.from_numpy(text_vec.astype(np.float32)).unsqueeze(0)  # (1, 384)
        x_stat = torch.from_numpy(stat_vec.astype(np.float32)).unsqueeze(0)  # (1, 25)

        with torch.no_grad():
            if self._fusion == "concat":
                fused = self._fusion_net(torch.cat([x_text, x_stat], dim=-1))
            else:
                fused = self._fusion_net(x_text, x_stat)

        out: np.ndarray = fused.squeeze(0).cpu().numpy()
        norm = np.linalg.norm(out)
        return out / norm if norm > 0 else out

    def embed_batch_descriptions(self, descriptions: list[str]) -> np.ndarray:
        """Encode a list of descriptions into an (N, 384) L2-normalised matrix."""
        vecs: np.ndarray = self.st_model.encode(
            descriptions, batch_size=32, convert_to_numpy=True
        )
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return vecs / norms

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_fusion(self, fusion: str, output_dim: int) -> nn.Module:
        """Construct and return the fusion network for the requested *fusion* strategy."""
        if fusion == "concat":
            return nn.Sequential(
                nn.Linear(_TEXT_DIM + _STAT_DIM, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim),
            )
        if fusion == "add":
            return _AddFusion(_TEXT_DIM, _STAT_DIM, output_dim)
        return _AttentionFusion(_TEXT_DIM, _STAT_DIM, output_dim)
