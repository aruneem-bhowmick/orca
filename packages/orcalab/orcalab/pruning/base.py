"""Abstract base class for trial pruning strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Pruner(ABC):
    """Abstract base class for trial pruning strategies.

    A pruner is called after each reported step of a trial and decides whether
    the trial should be stopped early.  Higher metric values are considered
    better (i.e. implementations should treat ``current_value`` as a
    maximisation objective).

    Implementations must be safe to call from a single thread.  No thread-safety
    guarantees are provided by this base class; concurrent access to shared
    internal state (e.g. ``ASHAPruner._promoted``) is the responsibility of the
    caller.
    """

    @abstractmethod
    def should_prune(
        self,
        trial_id: str,
        step: int,
        current_value: float,
        all_trial_values: dict[str, list[float]],
    ) -> bool:
        """Determine whether to prune a trial at the current step.

        Args:
            trial_id: Unique identifier of the trial being evaluated.
            step: Current training step or epoch (1-indexed; step ``k`` means
                ``k`` complete evaluations have been recorded).
            current_value: Performance metric value at ``step``.  Higher values
                are treated as better.
            all_trial_values: Mapping from trial identifier to the ordered list
                of metric values recorded so far.
                ``all_trial_values[tid][i]`` is the value at step ``i + 1`` for
                trial ``tid``.  The current trial may or may not be present; if
                present its list reflects values from *previous* steps only (the
                current step value is passed separately via ``current_value``).

        Returns:
            ``True`` if the trial should be stopped early, ``False`` to let it
            continue.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this pruning strategy."""
        ...
