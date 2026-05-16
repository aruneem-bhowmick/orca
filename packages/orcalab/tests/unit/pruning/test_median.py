"""Unit tests for MedianStoppingPruner."""

from __future__ import annotations

import pytest

from orcalab.pruning.base import Pruner
from orcalab.pruning.median import MedianStoppingPruner


class TestMedianPrunerABCCompliance:
    def test_is_pruner_instance(self) -> None:
        assert isinstance(MedianStoppingPruner(), Pruner)

    def test_name_property(self) -> None:
        assert MedianStoppingPruner().name == "median_stopping"

    def test_negative_warmup_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="warmup_steps"):
            MedianStoppingPruner(warmup_steps=-1)

    def test_zero_warmup_steps_valid(self) -> None:
        pruner = MedianStoppingPruner(warmup_steps=0)
        assert pruner.name == "median_stopping"


class TestMedianPrunerWarmup:
    def test_steps_before_warmup_never_pruned(self) -> None:
        pruner = MedianStoppingPruner(warmup_steps=5)
        peers = {"t1": [0.9, 0.9, 0.9, 0.9], "t2": [0.8, 0.8, 0.8, 0.8]}
        for step in range(1, 5):
            result = pruner.should_prune("t0", step, 0.01, peers)
            assert result is False, f"should not prune at step {step} (warmup=5)"

    def test_can_prune_at_warmup_step(self) -> None:
        pruner = MedianStoppingPruner(warmup_steps=5)
        peers = {"t1": [0.9] * 5, "t2": [0.8] * 5}
        result = pruner.should_prune("t0", 5, 0.01, peers)
        assert result is True

    def test_can_prune_after_warmup_step(self) -> None:
        pruner = MedianStoppingPruner(warmup_steps=3)
        peers = {"t1": [0.9] * 10, "t2": [0.8] * 10}
        result = pruner.should_prune("t0", 7, 0.1, peers)
        assert result is True

    def test_zero_warmup_prunes_at_step_1(self) -> None:
        pruner = MedianStoppingPruner(warmup_steps=0)
        peers = {"t1": [0.9], "t2": [0.8]}
        result = pruner.should_prune("t0", 1, 0.01, peers)
        assert result is True


class TestMedianPrunerLogic:
    def test_prunes_when_below_median(self) -> None:
        pruner = MedianStoppingPruner(warmup_steps=3)
        # Peer best values: t1→0.8, t2→0.6, t3→0.9; median = 0.8
        peers = {
            "t1": [0.6, 0.7, 0.8],
            "t2": [0.4, 0.5, 0.6],
            "t3": [0.7, 0.8, 0.9],
        }
        result = pruner.should_prune("t0", 3, 0.5, peers)
        assert result is True

    def test_no_prune_when_above_median(self) -> None:
        pruner = MedianStoppingPruner(warmup_steps=3)
        peers = {
            "t1": [0.6, 0.7, 0.8],
            "t2": [0.4, 0.5, 0.6],
            "t3": [0.7, 0.8, 0.9],
        }
        result = pruner.should_prune("t0", 3, 0.85, peers)
        assert result is False

    def test_no_prune_equal_to_median(self) -> None:
        pruner = MedianStoppingPruner(warmup_steps=2)
        peers = {"t1": [0.8, 0.8], "t2": [0.8, 0.8]}
        result = pruner.should_prune("t0", 2, 0.8, peers)
        assert result is False

    def test_no_prune_when_no_peers(self) -> None:
        pruner = MedianStoppingPruner(warmup_steps=2)
        result = pruner.should_prune("t0", 5, 0.0, {})
        assert result is False

    def test_no_prune_when_all_peers_empty(self) -> None:
        pruner = MedianStoppingPruner(warmup_steps=2)
        result = pruner.should_prune("t0", 5, 0.0, {"t1": [], "t2": []})
        assert result is False

    def test_peer_with_fewer_steps_counted_by_available_values(self) -> None:
        pruner = MedianStoppingPruner(warmup_steps=1)
        # Peer only has 1 value; current trial is at step 3
        peers = {"t1": [0.9]}
        result = pruner.should_prune("t0", 3, 0.5, peers)
        assert result is True  # 0.5 < median([0.9]) = 0.9

    def test_current_trial_excluded_from_peer_comparison(self) -> None:
        pruner = MedianStoppingPruner(warmup_steps=1)
        # Include the current trial in all_trial_values — should be ignored
        all_values = {"t0": [0.5, 0.5, 0.5], "t1": [0.9, 0.9, 0.9]}
        result = pruner.should_prune("t0", 3, 0.5, all_values)
        assert result is True  # only t1 is peer; 0.5 < 0.9

    def test_uses_best_value_up_to_step(self) -> None:
        pruner = MedianStoppingPruner(warmup_steps=1)
        # Peer peaked at step 2 (value 0.9), then dropped; best up to step 3 = 0.9
        peers = {"t1": [0.5, 0.9, 0.3]}
        result = pruner.should_prune("t0", 3, 0.5, peers)
        assert result is True  # 0.5 < max([0.5, 0.9, 0.3]) = 0.9

    def test_median_of_two_peers(self) -> None:
        pruner = MedianStoppingPruner(warmup_steps=1)
        # Two peers: best = 0.6, 0.8; median = (0.6+0.8)/2 = 0.7
        peers = {"t1": [0.6], "t2": [0.8]}
        assert pruner.should_prune("t0", 1, 0.5, peers) is True   # 0.5 < 0.7
        assert pruner.should_prune("t0", 1, 0.75, peers) is False  # 0.75 >= 0.7


class TestMedianPrunerBoundaryConditions:
    def test_step_zero_with_zero_warmup_no_peers(self) -> None:
        """Step 0 with warmup_steps=0: condition `0 < 0` is False, but no peers → not pruned."""
        pruner = MedianStoppingPruner(warmup_steps=0)
        result = pruner.should_prune("t0", 0, 0.0, {})
        assert result is False  # no peer_bests → returns False

    def test_step_zero_with_zero_warmup_with_peers(self) -> None:
        """Step 0 with warmup_steps=0 and peers: values[:0] is empty → no peer_bests → not pruned."""
        pruner = MedianStoppingPruner(warmup_steps=0)
        # values[:0] = [] regardless of peer values, so peer_bests is empty
        result = pruner.should_prune("t0", 0, 0.0, {"t1": [0.9, 0.9]})
        assert result is False

    def test_five_peers_odd_median_is_middle_value(self) -> None:
        """Median of 5 sorted peer bests is the 3rd value."""
        pruner = MedianStoppingPruner(warmup_steps=1)
        # Peer bests: 0.1, 0.3, 0.5, 0.7, 0.9 → median = 0.5
        peers = {
            "t1": [0.1],
            "t2": [0.3],
            "t3": [0.5],
            "t4": [0.7],
            "t5": [0.9],
        }
        assert pruner.should_prune("t0", 1, 0.4, peers) is True   # 0.4 < 0.5
        assert pruner.should_prune("t0", 1, 0.5, peers) is False  # equal → not pruned
        assert pruner.should_prune("t0", 1, 0.6, peers) is False  # 0.6 > 0.5

    def test_peer_values_sliced_beyond_length_uses_all_values(self) -> None:
        """values[:step] where step > len(values) safely returns all values."""
        pruner = MedianStoppingPruner(warmup_steps=1)
        # Peer has 2 values but step=10; values[:10] = [0.3, 0.9]
        peers = {"t1": [0.3, 0.9]}
        # best of [0.3, 0.9] = 0.9; current_value=0.5 < 0.9 → pruned
        result = pruner.should_prune("t0", 10, 0.5, peers)
        assert result is True

    def test_only_trial_in_dict_is_current_trial_no_prune(self) -> None:
        """If all_trial_values only contains the current trial itself, no peers → not pruned."""
        pruner = MedianStoppingPruner(warmup_steps=0)
        result = pruner.should_prune("t0", 5, 0.0, {"t0": [0.9, 0.9, 0.9, 0.9, 0.9]})
        assert result is False

    def test_warmup_boundary_exact_step_equals_warmup(self) -> None:
        """step == warmup_steps is NOT blocked by the warmup guard (`step < warmup_steps` is False)."""
        pruner = MedianStoppingPruner(warmup_steps=7)
        peers = {"t1": [0.9] * 7}
        # step=7 is not < 7, so median is computed; 0.01 < 0.9 → pruned
        result = pruner.should_prune("t0", 7, 0.01, peers)
        assert result is True

    def test_all_peers_same_value_equal_is_not_pruned(self) -> None:
        """If current_value exactly equals the peer median, trial is NOT pruned (strict < check)."""
        pruner = MedianStoppingPruner(warmup_steps=1)
        peers = {"t1": [0.5], "t2": [0.5], "t3": [0.5]}
        result = pruner.should_prune("t0", 1, 0.5, peers)
        assert result is False  # 0.5 is not < 0.5
