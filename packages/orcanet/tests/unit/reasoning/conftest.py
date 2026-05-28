"""Shared fixtures for orcanet reasoning unit tests."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import torch

from orca_shared.schemas.task import Task


def _make_task(**overrides) -> Task:
    defaults = dict(
        task_id=uuid4(),
        name="test-task",
        domain="vision",
        task_type="classification",
        n_samples=1000,
        n_features=10,
        n_classes=3,
        created_at=datetime(2024, 1, 1),
        updated_at=datetime(2024, 1, 1),
    )
    defaults.update(overrides)
    return Task(**defaults)


@pytest.fixture
def sample_task() -> Task:
    return _make_task()


@pytest.fixture
def another_task() -> Task:
    return _make_task(name="finance-regression", domain="finance", task_type="regression")


@pytest.fixture
def mock_retriever(sample_task):
    retriever = MagicMock()
    retriever.retrieve_with_expanded_queries = AsyncMock(
        return_value=[(sample_task, 0.88, "vector similarity")]
    )
    return retriever


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed = MagicMock(return_value=torch.zeros(64))
    return embedder


@pytest.fixture
def mock_task_repository(sample_task, another_task):
    repo = MagicMock()
    task_map = {sample_task.task_id: sample_task, another_task.task_id: another_task}
    repo.get_by_id = AsyncMock(side_effect=lambda uid: task_map.get(uid))
    return repo


@pytest.fixture
def mock_transfer_strategy():
    from orcanet.transfer.types import TransferScore

    strategy = MagicMock()
    strategy.score_transfer = MagicMock(
        return_value=TransferScore(
            overall=0.75,
            layer_scores={"layer1": 0.8},
            recommended_layers=["layer1"],
            reasoning="High CKA alignment on shared layers",
        )
    )
    return strategy


@pytest.fixture
def mock_orcamind_client():
    from orca_shared.schemas.metrics import PerformanceMetrics

    client = MagicMock()
    client.predict_performance = AsyncMock(
        return_value=PerformanceMetrics(
            experiment_id=uuid4(),
            final_metrics={"predicted_score": 0.82, "confidence": 0.9},
        )
    )
    return client
