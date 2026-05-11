"""Task sampling strategies for meta-training loops."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable

from orcamind.core.base import Task


class UniformTaskSampler:
    """Sample tasks uniformly at random without replacement."""

    def sample(self, task_pool: list[Task], n: int) -> list[Task]:
        if n > len(task_pool):
            raise ValueError(
                f"Cannot sample {n} tasks from pool of size {len(task_pool)}"
            )
        return random.sample(task_pool, n)


class CurriculumTaskSampler:
    """Sample tasks in order of increasing difficulty as training progresses.

    During the warmup period, only the easiest tasks are eligible. The eligible
    fraction grows linearly until all tasks are included at ``warmup_epochs``.
    """

    def __init__(
        self,
        difficulty_fn: Callable[[Task], float],
        warmup_epochs: int = 5,
    ) -> None:
        self._difficulty_fn = difficulty_fn
        self._warmup_epochs = max(1, warmup_epochs)

    def sample(self, task_pool: list[Task], n: int, epoch: int = 0) -> list[Task]:
        if n > len(task_pool):
            raise ValueError(
                f"Cannot sample {n} tasks from pool of size {len(task_pool)}"
            )
        sorted_tasks = sorted(task_pool, key=self._difficulty_fn)
        fraction = min(1.0, (epoch + 1) / self._warmup_epochs)
        eligible_count = max(n, math.ceil(len(sorted_tasks) * fraction))
        eligible = sorted_tasks[:eligible_count]
        return random.sample(eligible, n)


class DomainBalancedSampler:
    """Sample equal numbers of tasks from each domain, oversampling small domains.

    Args:
        domain_labels: Parallel list mapping ``task_pool[i]`` to its domain string.
    """

    def __init__(self, domain_labels: list[str]) -> None:
        self._domain_labels = domain_labels

    def sample(self, task_pool: list[Task], n: int) -> list[Task]:
        if len(task_pool) != len(self._domain_labels):
            raise ValueError(
                f"task_pool length ({len(task_pool)}) must match "
                f"domain_labels length ({len(self._domain_labels)})"
            )
        if n == 0:
            return []

        buckets: dict[str, list[Task]] = defaultdict(list)
        for task, label in zip(task_pool, self._domain_labels, strict=True):
            buckets[label].append(task)

        domains = sorted(buckets)
        n_domains = len(domains)
        per_domain = math.ceil(n / n_domains)

        result: list[Task] = []
        for domain in domains:
            pool = buckets[domain]
            if len(pool) >= per_domain:
                result.extend(random.sample(pool, per_domain))
            else:
                result.extend(random.choices(pool, k=per_domain))

        random.shuffle(result)
        return result[:n]
