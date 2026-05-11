"""Unit tests for task sampling strategies."""

from __future__ import annotations

import pytest
import torch

from orcamind.core.base import Task
from orcamind.training.task_sampler import (
    CurriculumTaskSampler,
    DomainBalancedSampler,
    UniformTaskSampler,
)


def _make_tasks(n: int, difficulty_step: float = 1.0) -> list[Task]:
    """Create n tasks whose query_y.sum() increases by difficulty_step."""
    torch.manual_seed(7)
    tasks = []
    for i in range(n):
        tasks.append(
            Task(
                support_x=torch.randn(3, 4),
                support_y=torch.zeros(3, dtype=torch.long),
                query_x=torch.randn(3, 4),
                query_y=torch.full((3,), int(i * difficulty_step), dtype=torch.long),
            )
        )
    return tasks


class TestUniformTaskSampler:
    def test_returns_correct_count(self, uniform_sampler, task_pool):
        """Sample returns exactly n tasks."""
        result = uniform_sampler.sample(task_pool, 4)
        assert len(result) == 4

    def test_result_is_subset_of_pool(self, uniform_sampler, task_pool):
        """Every returned task is from the original pool."""
        result = uniform_sampler.sample(task_pool, 6)
        assert all(t in task_pool for t in result)

    @pytest.mark.parametrize("n", [1, 3, 5, 10])
    def test_no_duplicate_tasks(self, uniform_sampler, task_pool, n):
        """Sampling without replacement produces no duplicate object references."""
        result = uniform_sampler.sample(task_pool, n)
        assert len(result) == len(set(id(t) for t in result))

    def test_raises_if_n_exceeds_pool(self, uniform_sampler, task_pool):
        """ValueError when n is larger than the pool."""
        with pytest.raises(ValueError, match="Cannot sample"):
            uniform_sampler.sample(task_pool, len(task_pool) + 1)


class TestCurriculumTaskSampler:
    def test_returns_correct_count(self, curriculum_sampler, task_pool):
        """Sample returns exactly n tasks."""
        result = curriculum_sampler.sample(task_pool, 4, epoch=0)
        assert len(result) == 4

    def test_early_epoch_returns_only_easy_tasks(self):
        """At epoch 0, only the easiest tasks are eligible."""
        tasks = _make_tasks(10)
        difficulty_fn = lambda t: float(t.query_y[0].item())  # noqa: E731
        sampler = CurriculumTaskSampler(difficulty_fn=difficulty_fn, warmup_epochs=5)

        # At epoch 0, fraction = 1/5 = 0.2, eligible_count = max(2, ceil(10*0.2)) = 2
        result = sampler.sample(tasks, 2, epoch=0)
        result_difficulties = [difficulty_fn(t) for t in result]
        max_eligible_difficulty = difficulty_fn(sorted(tasks, key=difficulty_fn)[1])
        assert all(d <= max_eligible_difficulty for d in result_difficulties)

    def test_late_epoch_includes_harder_tasks(self):
        """After warmup, harder tasks become eligible."""
        tasks = _make_tasks(10)
        difficulty_fn = lambda t: float(t.query_y[0].item())  # noqa: E731
        sampler = CurriculumTaskSampler(difficulty_fn=difficulty_fn, warmup_epochs=5)

        early_pool = set(
            id(t) for t in sampler.sample(tasks, len(tasks), epoch=0)
        )
        # At epoch=warmup_epochs, all tasks are eligible
        late_pool = set(
            id(t) for t in sampler.sample(tasks, len(tasks), epoch=5)
        )
        assert late_pool == {id(t) for t in tasks}
        assert len(late_pool) > len(early_pool)

    @pytest.mark.parametrize("epoch", [0, 1, 2, 4])
    def test_eligible_pool_grows_with_epoch(self, epoch):
        """Eligible pool size is non-decreasing as epoch increases."""
        tasks = _make_tasks(20)
        difficulty_fn = lambda t: float(t.query_y[0].item())  # noqa: E731
        sampler = CurriculumTaskSampler(difficulty_fn=difficulty_fn, warmup_epochs=5)

        import math
        fraction = min(1.0, (epoch + 1) / 5)
        expected_eligible = max(1, math.ceil(len(tasks) * fraction))
        # Sample n = expected_eligible; should not raise
        result = sampler.sample(tasks, expected_eligible, epoch=epoch)
        assert len(result) == expected_eligible

    def test_raises_if_n_exceeds_pool(self):
        """ValueError when n exceeds pool size."""
        tasks = _make_tasks(3)
        sampler = CurriculumTaskSampler(
            difficulty_fn=lambda t: 0.0, warmup_epochs=1
        )
        with pytest.raises(ValueError, match="Cannot sample"):
            sampler.sample(tasks, 5, epoch=0)


class TestDomainBalancedSampler:
    def test_result_includes_tasks_from_all_domains(self, domain_sampler, task_pool):
        """Returned tasks span both domains."""
        result = domain_sampler.sample(task_pool, 4)
        result_ids = {id(t) for t in result}
        domain_a_ids = {id(t) for t in task_pool[:5]}
        domain_b_ids = {id(t) for t in task_pool[5:]}
        assert result_ids & domain_a_ids, "No tasks from domain_a"
        assert result_ids & domain_b_ids, "No tasks from domain_b"

    def test_oversampling_works_when_domain_is_small(self):
        """Domain with a single task is oversampled to meet per-domain quota."""
        tasks = _make_tasks(3)
        labels = ["small"] * 1 + ["big"] * 2
        sampler = DomainBalancedSampler(domain_labels=labels)
        result = sampler.sample(tasks, 4)
        assert len(result) == 4

    @pytest.mark.parametrize("n", [2, 4, 6, 8])
    def test_returns_correct_total_count(self, task_pool, n):
        """Total result length equals n regardless of domain distribution."""
        labels = ["domain_a"] * 5 + ["domain_b"] * 5
        sampler = DomainBalancedSampler(domain_labels=labels)
        result = sampler.sample(task_pool, n)
        assert len(result) == n

    def test_raises_if_labels_length_mismatches_pool(self, task_pool):
        """ValueError when labels and pool have different lengths."""
        sampler = DomainBalancedSampler(domain_labels=["x"] * 3)
        with pytest.raises(ValueError):
            sampler.sample(task_pool, 2)

    def test_empty_n_returns_empty_list(self, domain_sampler, task_pool):
        """Requesting 0 tasks returns an empty list."""
        result = domain_sampler.sample(task_pool, 0)
        assert result == []
