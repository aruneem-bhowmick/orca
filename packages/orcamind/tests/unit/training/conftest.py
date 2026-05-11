"""Shared fixtures for training module unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from orcamind.core.base import Task
from orcamind.training.task_sampler import (
    CurriculumTaskSampler,
    DomainBalancedSampler,
    UniformTaskSampler,
)

_BATCH = 5
_FEATURES = 4
_CLASSES = 2
_N_TASKS = 10


@pytest.fixture()
def tiny_model() -> nn.Module:
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(_FEATURES, 8), nn.ReLU(), nn.Linear(8, _CLASSES)
    )


@pytest.fixture()
def task_pool() -> list[Task]:
    torch.manual_seed(0)
    tasks = []
    for _ in range(_N_TASKS):
        tasks.append(
            Task(
                support_x=torch.randn(_BATCH, _FEATURES),
                support_y=torch.randint(0, _CLASSES, (_BATCH,)),
                query_x=torch.randn(_BATCH, _FEATURES),
                query_y=torch.randint(0, _CLASSES, (_BATCH,)),
            )
        )
    return tasks


@pytest.fixture()
def mock_meta_learner(tiny_model: nn.Module) -> MagicMock:
    m = MagicMock()
    m.meta_update.return_value = {
        "meta_train_loss": 0.5,
        "meta_train_accuracy": 0.7,
    }
    m.adapt.return_value = tiny_model
    m.evaluate_task.return_value = {"loss": 0.4, "accuracy": 0.8}
    return m


@pytest.fixture()
def uniform_sampler() -> UniformTaskSampler:
    return UniformTaskSampler()


@pytest.fixture()
def curriculum_sampler() -> CurriculumTaskSampler:
    def difficulty_fn(task: Task) -> float:
        return float(task.query_y.sum().item())

    return CurriculumTaskSampler(difficulty_fn=difficulty_fn, warmup_epochs=5)


@pytest.fixture()
def domain_sampler(task_pool: list[Task]) -> DomainBalancedSampler:
    labels = ["domain_a"] * 5 + ["domain_b"] * 5
    return DomainBalancedSampler(domain_labels=labels)
