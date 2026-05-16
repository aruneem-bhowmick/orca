"""Unit tests for ASHAPruner."""

from __future__ import annotations

import pytest

from orcalab.pruning.asha import ASHAPruner
from orcalab.pruning.base import Pruner


class TestASHAPrunerABCCompliance:
    def test_is_pruner_instance(self) -> None:
        assert isinstance(ASHAPruner(), Pruner)

    def test_name_property(self) -> None:
        assert ASHAPruner().name == "asha"

    def test_invalid_min_resource_raises(self) -> None:
        with pytest.raises(ValueError, match="min_resource"):
            ASHAPruner(min_resource=0)

    def test_invalid_reduction_factor_raises(self) -> None:
        with pytest.raises(ValueError, match="reduction_factor"):
            ASHAPruner(reduction_factor=1)

    def test_max_less_than_min_raises(self) -> None:
        with pytest.raises(ValueError):
            ASHAPruner(min_resource=10, max_resource=5)

    def test_default_params_accepted(self) -> None:
        pruner = ASHAPruner()
        assert pruner.name == "asha"


class TestASHAPrunerRungDetection:
    def test_non_rung_step_never_pruned(self) -> None:
        pruner = ASHAPruner(min_resource=1, max_resource=27, reduction_factor=3)
        # Rungs at 1, 3, 9, 27; steps 2, 4, 5 are not rung levels
        all_values = {f"t{i}": [float(i)] * 5 for i in range(10)}
        for step in (2, 4, 5):
            result = pruner.should_prune("t0", step, 0.0, all_values)
            assert result is False, f"step {step} is not a rung; must not prune"

    def test_single_trial_never_pruned_at_rung(self) -> None:
        pruner = ASHAPruner(min_resource=1, max_resource=27, reduction_factor=3)
        result = pruner.should_prune("t0", 1, 0.0, {})
        assert result is False

    def test_step_beyond_max_resource_not_a_rung(self) -> None:
        pruner = ASHAPruner(min_resource=1, max_resource=9, reduction_factor=3)
        # Max rung is at step 9; step 27 is outside max_resource
        all_values = {"t1": [0.9] * 27}
        result = pruner.should_prune("t0", 27, 0.0, all_values)
        assert result is False

    def test_rung_steps_are_correct(self) -> None:
        pruner = ASHAPruner(min_resource=1, max_resource=27, reduction_factor=3)
        rungs = pruner._rungs()
        steps = [step for _, step in rungs]
        assert steps == [1, 3, 9, 27]

    def test_custom_min_resource_rung_steps(self) -> None:
        pruner = ASHAPruner(min_resource=2, max_resource=54, reduction_factor=3)
        rungs = pruner._rungs()
        steps = [step for _, step in rungs]
        assert steps == [2, 6, 18, 54]


class TestASHAPrunerPruningRatio:
    def test_bottom_two_thirds_pruned_at_first_rung_9_trials(self) -> None:
        """With 9 trials and reduction_factor=3, exactly 6/9 = 2/3 are pruned.

        All trial values are pre-filled at step 1 so every trial competes against all others.
        """
        pruner = ASHAPruner(min_resource=1, max_resource=81, reduction_factor=3)
        n = 9
        all_values = {f"t{i}": [float(i) / n] for i in range(n)}

        pruned = [
            f"t{i}"
            for i in range(n)
            if pruner.should_prune(f"t{i}", 1, float(i) / n, all_values)
        ]
        assert len(pruned) == 6

    def test_top_three_of_nine_not_pruned(self) -> None:
        pruner = ASHAPruner(min_resource=1, max_resource=81, reduction_factor=3)
        n = 9
        all_values = {f"t{i}": [float(i) / n] for i in range(n)}

        for i in range(6, 9):
            result = pruner.should_prune(f"t{i}", 1, float(i) / n, all_values)
            assert result is False, f"t{i} is in the top-3 and must not be pruned"

    def test_promoted_dict_updated_for_survivors(self) -> None:
        pruner = ASHAPruner(min_resource=1, max_resource=9, reduction_factor=3)
        all_values = {"t0": [0.1], "t1": [0.5], "t2": [0.9]}
        pruner.should_prune("t2", 1, 0.9, all_values)  # best → promoted
        assert 0 in pruner._promoted
        assert "t2" in pruner._promoted[0]

    def test_pruned_trial_not_in_promoted(self) -> None:
        pruner = ASHAPruner(min_resource=1, max_resource=9, reduction_factor=3)
        all_values = {"t0": [0.1], "t1": [0.5], "t2": [0.9]}
        pruner.should_prune("t0", 1, 0.1, all_values)  # worst → pruned
        promoted_all = [tid for ids in pruner._promoted.values() for tid in ids]
        assert "t0" not in promoted_all


class TestASHAPrunerTopTrialSafety:
    def test_best_trial_not_pruned_at_any_rung(self) -> None:
        """Best-quality trial must not be pruned at any rung when full history is available."""
        pruner = ASHAPruner(min_resource=1, max_resource=81, reduction_factor=3)
        n = 9
        # Pre-fill all_values so every trial has a complete history up to max_resource
        all_values = {f"t{i}": [float(i + 1) / n] * 81 for i in range(n)}
        best_tid = f"t{n - 1}"
        best_val = float(n) / n  # = 1.0

        for _, step in pruner._rungs():
            result = pruner.should_prune(best_tid, step, best_val, all_values)
            assert result is False, f"Best trial pruned at step {step}"

    def test_best_trial_survives_sequential_simulation(self) -> None:
        """Best trial must survive when trials are run sequentially, best-quality first."""
        pruner = ASHAPruner(min_resource=1, max_resource=27, reduction_factor=3)
        n = 9
        qualities = {f"t{i}": (i + 1) / n for i in range(n)}
        all_values: dict[str, list[float]] = {tid: [] for tid in qualities}
        best_tid = f"t{n - 1}"

        for tid in sorted(qualities, key=lambda t: -qualities[t]):
            q = qualities[tid]
            for step in range(1, 28):
                if pruner.should_prune(tid, step, q, all_values):
                    assert tid != best_tid, f"Best trial {best_tid!r} pruned at step {step}"
                    break
                all_values[tid].append(q)


class TestASHAPrunerComputeSavings:
    def test_savings_at_least_40_percent_vs_no_pruning(self) -> None:
        """ASHA must execute ≤60% of steps compared to running all trials to max_resource.

        Simulation: trials run sequentially, best-quality first. The best trial completes
        all steps first; every subsequent (worse) trial is pruned at the first rung because
        it cannot beat the best trial's established record.
        """
        pruner = ASHAPruner(min_resource=1, max_resource=81, reduction_factor=3)
        n_trials = 20
        max_steps = 81

        qualities = {f"trial_{i}": (i + 1) / n_trials for i in range(n_trials)}
        all_values: dict[str, list[float]] = {tid: [] for tid in qualities}
        total_steps = 0

        for tid in sorted(qualities, key=lambda t: -qualities[t]):
            q = qualities[tid]
            for step in range(1, max_steps + 1):
                total_steps += 1  # trial executed this step
                if pruner.should_prune(tid, step, q, all_values):
                    break
                all_values[tid].append(q)

        no_pruning_steps = n_trials * max_steps  # 1620
        savings = 1.0 - total_steps / no_pruning_steps
        assert savings >= 0.40, (
            f"Expected ≥40% compute savings, got {savings:.1%} "
            f"({total_steps} steps vs {no_pruning_steps} without pruning)"
        )


class TestASHAPrunerRungForStep:
    """Direct unit tests for the _rung_for_step helper."""

    def test_non_rung_step_returns_none(self) -> None:
        pruner = ASHAPruner(min_resource=1, max_resource=27, reduction_factor=3)
        # Rungs are at 1, 3, 9, 27; step 2 is not a rung
        assert pruner._rung_for_step(2) is None

    def test_rung_step_returns_correct_index(self) -> None:
        pruner = ASHAPruner(min_resource=1, max_resource=27, reduction_factor=3)
        # step 1 → rung_idx 0, step 3 → 1, step 9 → 2, step 27 → 3
        assert pruner._rung_for_step(1) == 0
        assert pruner._rung_for_step(3) == 1
        assert pruner._rung_for_step(9) == 2
        assert pruner._rung_for_step(27) == 3

    def test_step_zero_is_not_a_rung(self) -> None:
        pruner = ASHAPruner(min_resource=1, max_resource=27, reduction_factor=3)
        assert pruner._rung_for_step(0) is None

    def test_step_beyond_max_resource_returns_none(self) -> None:
        pruner = ASHAPruner(min_resource=1, max_resource=9, reduction_factor=3)
        assert pruner._rung_for_step(27) is None


class TestASHAPrunerBoundaryConditions:
    def test_min_resource_equals_max_resource_single_rung(self) -> None:
        """When min_resource == max_resource there is exactly one rung."""
        pruner = ASHAPruner(min_resource=5, max_resource=5, reduction_factor=3)
        rungs = pruner._rungs()
        assert len(rungs) == 1
        assert rungs[0] == (0, 5)

    def test_min_resource_equals_max_resource_prunes_correctly(self) -> None:
        """With one rung and 3 trials (rf=3), only the best 1 survives."""
        pruner = ASHAPruner(min_resource=5, max_resource=5, reduction_factor=3)
        all_values = {"t0": [0.1] * 5, "t1": [0.5] * 5, "t2": [0.9] * 5}
        # worst trial (t0) must be pruned
        assert pruner.should_prune("t0", 5, 0.1, all_values) is True
        # best trial (t2) must survive
        assert pruner.should_prune("t2", 5, 0.9, all_values) is False

    def test_reduction_factor_2_boundary_accepted(self) -> None:
        """reduction_factor=2 is the minimum valid value."""
        pruner = ASHAPruner(min_resource=1, max_resource=8, reduction_factor=2)
        rungs = pruner._rungs()
        steps = [step for _, step in rungs]
        assert steps == [1, 2, 4, 8]

    def test_reduction_factor_2_keeps_half(self) -> None:
        """With rf=2 and 4 trials, top 2 survive at first rung."""
        pruner = ASHAPruner(min_resource=1, max_resource=8, reduction_factor=2)
        all_values = {f"t{i}": [float(i)] for i in range(4)}
        pruned = [
            f"t{i}"
            for i in range(4)
            if pruner.should_prune(f"t{i}", 1, float(i), all_values)
        ]
        assert len(pruned) == 2  # bottom half pruned

    def test_trial_with_too_few_values_excluded_from_rung(self) -> None:
        """A peer with fewer than `step` values is not counted at that rung."""
        pruner = ASHAPruner(min_resource=3, max_resource=27, reduction_factor=3)
        # Rung is at step 3; t1 has only 2 values (< 3) → not included
        all_values = {"t1": [0.9, 0.9]}  # only 2 values, step=3
        # With just the current trial in competition, keep=max(1,1//3)=1, current always kept
        result = pruner.should_prune("t0", 3, 0.01, all_values)
        assert result is False  # t0 is the only eligible trial → never pruned

    def test_promoted_dict_idempotent_for_repeated_calls(self) -> None:
        """Calling should_prune twice for the same winner at the same rung
        must not duplicate the trial_id in _promoted."""
        pruner = ASHAPruner(min_resource=1, max_resource=9, reduction_factor=3)
        all_values = {"t0": [0.1], "t1": [0.5], "t2": [0.9]}
        pruner.should_prune("t2", 1, 0.9, all_values)
        pruner.should_prune("t2", 1, 0.9, all_values)  # same call again
        assert pruner._promoted[0].count("t2") == 1  # must appear exactly once

    def test_negative_min_resource_raises(self) -> None:
        with pytest.raises(ValueError, match="min_resource"):
            ASHAPruner(min_resource=-5)

    def test_promoted_trial_not_re_pruned_when_stronger_peer_arrives(self) -> None:
        """A trial promoted at a rung must not be pruned on re-evaluation even if a
        stronger peer has since arrived and would push it out of the top-k."""
        pruner = ASHAPruner(min_resource=1, max_resource=9, reduction_factor=3)
        # t0 is the only trial at rung 0 → keep=max(1,1//3)=1 → promoted
        assert pruner.should_prune("t0", 1, 0.5, {}) is False
        assert "t0" in pruner._promoted.get(0, [])
        # A strictly better peer now appears in all_trial_values
        # Without the sticky-promotion guard, fresh re-ranking would prune t0
        assert pruner.should_prune("t0", 1, 0.5, {"t1": [1.0]}) is False

    def test_tied_values_at_rung_keep_at_least_one(self) -> None:
        """When all trials share the same value at a rung, at least one must survive."""
        pruner = ASHAPruner(min_resource=1, max_resource=9, reduction_factor=3)
        # 3 trials with identical values
        all_values = {"t0": [0.5], "t1": [0.5], "t2": [0.5]}
        results = [
            pruner.should_prune(f"t{i}", 1, 0.5, all_values) for i in range(3)
        ]
        assert results.count(False) >= 1  # at least one trial is not pruned
