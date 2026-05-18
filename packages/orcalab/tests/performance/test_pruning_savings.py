"""Performance benchmark: ASHA pruner compute-savings assertion.

Simulates 20 trials on a concave (quadratic) training objective and
measures how many total training steps ASHA executes compared with
running every trial to completion.  The target is ≥40% savings.

Each trial has a fixed "quality" q ∈ (0, 1].  Its metric at epoch t is
modelled as a concave quadratic: metric(t) = q * (1 - (1 - t/T)²),
which is monotonically increasing in t and in q, mirroring real
learning curves where high-quality trials converge faster/higher.
"""

from __future__ import annotations

import pytest

from orcalab.pruning.asha import ASHAPruner


N_TRIALS = 20
MAX_RESOURCE = 81  # epochs (= 3^4, so rungs at 1, 3, 9, 27, 81)
MIN_RESOURCE = 1
REDUCTION_FACTOR = 3
TARGET_SAVINGS = 0.40


def _metric(quality: float, epoch: int, max_epoch: int = MAX_RESOURCE) -> float:
    """Concave quadratic learning curve; higher quality → higher final value."""
    progress = epoch / max_epoch
    return quality * (1.0 - (1.0 - progress) ** 2)


def _simulate(pruner: ASHAPruner) -> int:
    """Run all trials sequentially (best quality first) and return total steps executed."""
    qualities = {f"trial_{i}": (i + 1) / N_TRIALS for i in range(N_TRIALS)}
    all_values: dict[str, list[float]] = {tid: [] for tid in qualities}
    total_steps = 0

    for tid in sorted(qualities, key=lambda t: -qualities[t]):
        q = qualities[tid]
        for step in range(1, MAX_RESOURCE + 1):
            metric = _metric(q, step)
            total_steps += 1
            if pruner.should_prune(tid, step, metric, all_values):
                break
            all_values[tid].append(metric)

    return total_steps


class TestASHAPruningSavings:
    """Executable compute-savings assertions for the ASHA pruner.

    Each test drives a deterministic synthetic sweep — trials ordered by quality,
    metrics generated from a concave quadratic curve — and makes a measurable claim
    about ASHA's pruning behaviour that would catch a regression in the pruner logic.
    """

    def test_compute_savings_at_least_40_percent(self) -> None:
        """ASHA must execute ≤60% of the steps of an unpruned baseline."""
        pruner = ASHAPruner(
            min_resource=MIN_RESOURCE,
            max_resource=MAX_RESOURCE,
            reduction_factor=REDUCTION_FACTOR,
        )
        executed_steps = _simulate(pruner)
        baseline_steps = N_TRIALS * MAX_RESOURCE  # 1620

        savings = 1.0 - executed_steps / baseline_steps
        assert savings >= TARGET_SAVINGS, (
            f"Expected ≥{TARGET_SAVINGS:.0%} compute savings; "
            f"got {savings:.1%} ({executed_steps} / {baseline_steps} steps)"
        )

    def test_best_trial_completes_all_steps(self) -> None:
        """The best trial (quality=1.0) must never be pruned and run to max_resource."""
        pruner = ASHAPruner(
            min_resource=MIN_RESOURCE,
            max_resource=MAX_RESOURCE,
            reduction_factor=REDUCTION_FACTOR,
        )
        qualities = {f"trial_{i}": (i + 1) / N_TRIALS for i in range(N_TRIALS)}
        all_values: dict[str, list[float]] = {tid: [] for tid in qualities}

        best_tid = f"trial_{N_TRIALS - 1}"
        best_quality = 1.0
        steps_completed = 0

        for step in range(1, MAX_RESOURCE + 1):
            metric = _metric(best_quality, step)
            pruned = pruner.should_prune(best_tid, step, metric, all_values)
            assert not pruned, f"Best trial pruned at step {step}"
            all_values[best_tid].append(metric)
            steps_completed += 1

        assert steps_completed == MAX_RESOURCE

    def test_worst_trial_pruned_before_completion(self) -> None:
        """The worst trial (quality=1/N) must be pruned before running all max_resource steps."""
        pruner = ASHAPruner(
            min_resource=MIN_RESOURCE,
            max_resource=MAX_RESOURCE,
            reduction_factor=REDUCTION_FACTOR,
        )
        qualities = {f"trial_{i}": (i + 1) / N_TRIALS for i in range(N_TRIALS)}
        all_values: dict[str, list[float]] = {tid: [] for tid in qualities}

        worst_tid = "trial_0"
        worst_quality = 1 / N_TRIALS

        # First let the best trial run to completion so the worst one faces competition
        best_tid = f"trial_{N_TRIALS - 1}"
        for step in range(1, MAX_RESOURCE + 1):
            all_values[best_tid].append(_metric(1.0, step))

        pruned_at: int | None = None
        for step in range(1, MAX_RESOURCE + 1):
            metric = _metric(worst_quality, step)
            if pruner.should_prune(worst_tid, step, metric, all_values):
                pruned_at = step
                break
            all_values[worst_tid].append(metric)

        assert pruned_at is not None, "Worst trial should have been pruned"
        assert pruned_at < MAX_RESOURCE

    def test_savings_scale_with_more_trials(self) -> None:
        """Savings must meet TARGET_SAVINGS and must be ≥ the 20-trial baseline savings."""

        def _run_simulation(n: int) -> float:
            pruner = ASHAPruner(
                min_resource=MIN_RESOURCE,
                max_resource=MAX_RESOURCE,
                reduction_factor=REDUCTION_FACTOR,
            )
            qualities = {f"trial_{i}": (i + 1) / n for i in range(n)}
            all_values: dict[str, list[float]] = {tid: [] for tid in qualities}
            total_steps = 0
            for tid in sorted(qualities, key=lambda t: -qualities[t]):
                q = qualities[tid]
                for step in range(1, MAX_RESOURCE + 1):
                    total_steps += 1
                    metric = _metric(q, step)
                    if pruner.should_prune(tid, step, metric, all_values):
                        break
                    all_values[tid].append(metric)
            baseline = n * MAX_RESOURCE
            return 1.0 - total_steps / baseline

        savings_20 = _run_simulation(N_TRIALS)  # N_TRIALS = 20
        extended_n = 27
        savings = _run_simulation(extended_n)

        assert savings >= TARGET_SAVINGS, (
            f"Expected ≥{TARGET_SAVINGS:.0%} savings with {extended_n} trials; "
            f"got {savings:.1%}"
        )
        assert savings >= savings_20, (
            f"Expected savings with {extended_n} trials ({savings:.1%}) to be at least as good "
            f"as savings with {N_TRIALS} trials ({savings_20:.1%})"
        )
