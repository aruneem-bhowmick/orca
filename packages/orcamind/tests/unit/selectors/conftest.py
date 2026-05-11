"""Shared fixtures for selector unit tests."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from orca_shared.schemas.model import ModelConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_TASKS = 20
N_MODELS = 5
EMBED_DIM = 32
BEST_MODEL_IDX = 0
BEST_PERF = 0.9
OTHER_PERF = 0.1
SEED = 42


def _make_model_configs() -> list[ModelConfig]:
    """Five distinct models; model 0 has small hidden_size to be identifiable."""
    configs = []
    for i in range(N_MODELS):
        configs.append(
            ModelConfig(
                model_id=uuid4(),
                name=f"model_{i}",
                architecture=f"arch_{i}",
                config={
                    "hidden_size": 64 if i == BEST_MODEL_IDX else 256,
                    "num_layers": 2,
                    "lr": 1e-3,
                    "batch_size": 32,
                },
                parameter_count=100_000 * (i + 1),
                flops=1_000_000 * (i + 1),
            )
        )
    return configs


def _make_training_data(
    model_configs: list[ModelConfig],
) -> tuple[np.ndarray, list[ModelConfig], np.ndarray]:
    """100 (embedding, model_config, performance) triples.

    Each of N_TASKS tasks is paired with every model once.
    BEST_MODEL_IDX always receives BEST_PERF; others receive OTHER_PERF.
    """
    rng = np.random.default_rng(SEED)
    task_embeddings_raw = rng.standard_normal((N_TASKS, EMBED_DIM))

    embeddings: list[np.ndarray] = []
    configs: list[ModelConfig] = []
    perfs: list[float] = []

    for task_emb in task_embeddings_raw:
        for model_idx, model in enumerate(model_configs):
            embeddings.append(task_emb.copy())
            configs.append(model)
            perfs.append(BEST_PERF if model_idx == BEST_MODEL_IDX else OTHER_PERF)

    return (
        np.array(embeddings),
        configs,
        np.array(perfs, dtype=np.float64),
    )


@pytest.fixture(scope="session")
def model_configs() -> list[ModelConfig]:
    return _make_model_configs()


@pytest.fixture(scope="session")
def training_data(
    model_configs: list[ModelConfig],
) -> tuple[np.ndarray, list[ModelConfig], np.ndarray]:
    return _make_training_data(model_configs)


@pytest.fixture(scope="session")
def held_out_embeddings() -> np.ndarray:
    """Five embeddings not in the training set (perturbed from training distribution)."""
    rng = np.random.default_rng(SEED + 1)
    return rng.standard_normal((5, EMBED_DIM))
