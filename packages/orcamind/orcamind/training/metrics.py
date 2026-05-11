"""Meta-learning evaluation metrics."""

from __future__ import annotations

import math

import numpy as np

from orcamind.core.base import MetaLearner, Task


def k_shot_accuracy(
    model: MetaLearner,
    tasks: list[Task],
    k_shot: int,
    n_query: int,
) -> float:
    """Compute mean accuracy after k-shot adaptation across a list of tasks.

    NaN accuracy values (e.g., from regression learners) are skipped. Returns
    0.0 if every task yields NaN.
    """
    accuracies: list[float] = []
    for task in tasks:
        adapted = model.adapt(
            task.support_x[:k_shot],
            task.support_y[:k_shot],
        )
        result = model.evaluate_task(
            adapted,
            task.query_x[:n_query],
            task.query_y[:n_query],
        )
        acc = result.get("accuracy", float("nan"))
        if not math.isnan(acc):
            accuracies.append(acc)

    return float(np.mean(accuracies)) if accuracies else 0.0


def adaptation_efficiency(losses_by_step: list[list[float]]) -> float:
    """Area-under-curve efficiency score for an adaptation trajectory.

    Args:
        losses_by_step: ``losses_by_step[i]`` is the per-step loss list for
            task ``i``. All inner lists must have equal length.

    Returns:
        A score in [0, 1] where 1.0 means instantaneous adaptation (loss is
        already zero before any steps) and 0.0 means no adaptation at all.

    Raises:
        ValueError: If inner lists differ in length or the input is empty.
    """
    if not losses_by_step:
        raise ValueError("losses_by_step must not be empty")

    n_steps = len(losses_by_step[0])
    if n_steps == 0:
        raise ValueError(
            "each task loss trajectory must contain at least one step"
        )
    for i, steps in enumerate(losses_by_step):
        if len(steps) != n_steps:
            raise ValueError(
                f"All step lists must have equal length; "
                f"list 0 has {n_steps} steps but list {i} has {len(steps)}"
            )

    mean_curve = [
        float(np.mean([task_losses[s] for task_losses in losses_by_step]))
        for s in range(n_steps)
    ]

    if mean_curve[0] == 0.0:
        return 1.0

    normed = [min(1.0, max(0.0, v / mean_curve[0])) for v in mean_curve]

    if n_steps == 1:
        return 1.0 - normed[0]

    auc = float(np.trapezoid(normed)) / (n_steps - 1)
    return 1.0 - float(np.clip(auc, 0.0, 1.0))


def catastrophic_forgetting(
    model: MetaLearner,
    old_tasks: list[Task],
    new_tasks: list[Task],
    k_shot: int = 5,
    n_query: int = 15,
) -> float:
    """Measure the performance drop on old tasks after training on new ones.

    Mutates ``model`` in-place via ``meta_update``. Callers who need to
    preserve the original model state should deep-copy before calling.

    Returns:
        Non-negative float; larger values indicate more forgetting.
    """
    acc_before = k_shot_accuracy(model, old_tasks, k_shot, n_query)
    model.meta_update(new_tasks)
    acc_after = k_shot_accuracy(model, old_tasks, k_shot, n_query)
    return max(0.0, acc_before - acc_after)
