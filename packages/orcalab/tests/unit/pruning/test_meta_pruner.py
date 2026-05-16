"""Unit tests for MetaPruner."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from orca_shared.clients.orcamind_client import OrcaMindClient
from orca_shared.schemas.metrics import PerformanceMetrics
from orcalab.pruning.asha import ASHAPruner
from orcalab.pruning.base import Pruner
from orcalab.pruning.meta_pruner import MetaPruner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics(score: float) -> PerformanceMetrics:
    return PerformanceMetrics(
        experiment_id=uuid4(),
        final_metrics={"accuracy": score},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock(spec=OrcaMindClient)
    client.predict_performance = AsyncMock(return_value=_make_metrics(0.5))
    return client


@pytest.fixture
def asha_base() -> ASHAPruner:
    return ASHAPruner(min_resource=1, max_resource=81, reduction_factor=3)


# ---------------------------------------------------------------------------
# TestMetaPrunerABCCompliance
# ---------------------------------------------------------------------------


class TestMetaPrunerABCCompliance:
    def test_is_pruner_instance(self, mock_client: MagicMock, asha_base: ASHAPruner) -> None:
        assert isinstance(MetaPruner(mock_client, asha_base), Pruner)

    def test_name_property(self, mock_client: MagicMock, asha_base: ASHAPruner) -> None:
        assert MetaPruner(mock_client, asha_base).name == "meta_pruner"


# ---------------------------------------------------------------------------
# TestMetaPrunerMinSteps
# ---------------------------------------------------------------------------


class TestMetaPrunerMinSteps:
    def test_no_prune_below_min_steps(
        self, mock_client: MagicMock, asha_base: ASHAPruner
    ) -> None:
        pruner = MetaPruner(mock_client, asha_base, min_steps_before_prediction=10)
        for step in range(1, 10):
            result = pruner.should_prune("t0", step, 0.01, {})
            assert result is False, f"must not prune at step {step} (min_steps=10)"

    def test_no_prune_at_min_steps_minus_1(
        self, mock_client: MagicMock, asha_base: ASHAPruner
    ) -> None:
        pruner = MetaPruner(mock_client, asha_base, min_steps_before_prediction=10)
        result = pruner.should_prune("t0", 9, 0.01, {})
        assert result is False

    def test_orcamind_not_queried_before_min_steps(
        self, mock_client: MagicMock, asha_base: ASHAPruner
    ) -> None:
        pruner = MetaPruner(mock_client, asha_base, min_steps_before_prediction=10)
        pruner.should_prune("t0", 5, 0.01, {})
        mock_client.predict_performance.assert_not_called()


# ---------------------------------------------------------------------------
# TestMetaPrunerOrcaMindPruning
# ---------------------------------------------------------------------------


class TestMetaPrunerOrcaMindPruning:
    def test_prunes_when_prediction_below_threshold(
        self, mock_client: MagicMock, asha_base: ASHAPruner
    ) -> None:
        mock_client.predict_performance = AsyncMock(return_value=_make_metrics(0.1))
        pruner = MetaPruner(
            mock_client, asha_base, prediction_threshold=0.3, min_steps_before_prediction=5
        )
        result = pruner.should_prune("t0", 10, 0.2, {})
        assert result is True

    def test_no_prune_when_prediction_above_threshold(
        self, mock_client: MagicMock, asha_base: ASHAPruner
    ) -> None:
        mock_client.predict_performance = AsyncMock(return_value=_make_metrics(0.9))
        pruner = MetaPruner(
            mock_client, asha_base, prediction_threshold=0.3, min_steps_before_prediction=5
        )
        # OrcaMind says 0.9 > 0.3; ASHA has only one trial → not pruned either
        result = pruner.should_prune("t0", 10, 0.7, {})
        assert result is False

    def test_no_prune_when_prediction_equals_threshold(
        self, mock_client: MagicMock, asha_base: ASHAPruner
    ) -> None:
        mock_client.predict_performance = AsyncMock(return_value=_make_metrics(0.3))
        pruner = MetaPruner(
            mock_client, asha_base, prediction_threshold=0.3, min_steps_before_prediction=1
        )
        result = pruner.should_prune("t0", 5, 0.3, {})
        assert result is False  # strictly less than; equal is NOT pruned

    def test_orcamind_queried_at_or_above_min_steps(
        self, mock_client: MagicMock, asha_base: ASHAPruner
    ) -> None:
        pruner = MetaPruner(mock_client, asha_base, min_steps_before_prediction=10)
        pruner.should_prune("t0", 10, 0.5, {})
        mock_client.predict_performance.assert_called_once()


# ---------------------------------------------------------------------------
# TestMetaPrunerEarlyPruning
# ---------------------------------------------------------------------------


class TestMetaPrunerEarlyPruning:
    def test_prunes_before_base_pruner_would(self, mock_client: MagicMock) -> None:
        """OrcaMind low prediction triggers pruning even when base_pruner wouldn't."""
        base = MagicMock(spec=Pruner)
        base.should_prune = MagicMock(return_value=False)

        mock_client.predict_performance = AsyncMock(return_value=_make_metrics(0.05))
        pruner = MetaPruner(
            mock_client, base, prediction_threshold=0.3, min_steps_before_prediction=1
        )
        result = pruner.should_prune("t0", 5, 0.5, {})
        assert result is True
        base.should_prune.assert_not_called()

    def test_base_pruner_consulted_when_prediction_is_safe(
        self, mock_client: MagicMock
    ) -> None:
        base = MagicMock(spec=Pruner)
        base.should_prune = MagicMock(return_value=True)

        mock_client.predict_performance = AsyncMock(return_value=_make_metrics(0.9))
        pruner = MetaPruner(
            mock_client, base, prediction_threshold=0.3, min_steps_before_prediction=1
        )
        result = pruner.should_prune("t0", 5, 0.5, {})
        assert result is True
        base.should_prune.assert_called_once()

    def test_base_pruner_result_returned_when_prediction_safe(
        self, mock_client: MagicMock
    ) -> None:
        base = MagicMock(spec=Pruner)
        base.should_prune = MagicMock(return_value=False)

        mock_client.predict_performance = AsyncMock(return_value=_make_metrics(0.9))
        pruner = MetaPruner(
            mock_client, base, prediction_threshold=0.3, min_steps_before_prediction=1
        )
        result = pruner.should_prune("t0", 5, 0.5, {})
        assert result is False
        base.should_prune.assert_called_once()


# ---------------------------------------------------------------------------
# TestMetaPrunerFallback
# ---------------------------------------------------------------------------


class TestMetaPrunerFallback:
    def test_falls_back_to_base_pruner_on_exception(self) -> None:
        """When OrcaMind raises, _query_orcamind returns None and base_pruner decides."""
        bad_client = MagicMock(spec=OrcaMindClient)
        bad_client.predict_performance = AsyncMock(side_effect=Exception("timeout"))

        base = MagicMock(spec=Pruner)
        base.should_prune = MagicMock(return_value=True)

        pruner = MetaPruner(bad_client, base, min_steps_before_prediction=1)
        result = pruner.should_prune("t0", 5, 0.5, {})
        assert result is True
        base.should_prune.assert_called_once()

    def test_bottom_quality_trial_pruned_by_asha_fallback(self) -> None:
        """Unreachable OrcaMind → fallback to ASHAPruner which prunes low-quality trial."""
        bad_client = MagicMock(spec=OrcaMindClient)
        bad_client.predict_performance = AsyncMock(side_effect=Exception("refused"))

        asha = ASHAPruner(min_resource=1, max_resource=81, reduction_factor=3)
        pruner = MetaPruner(bad_client, asha, min_steps_before_prediction=1)

        n = 9
        all_values = {f"t{i}": [float(i) / n] for i in range(n)}
        result = pruner.should_prune("t0", 1, 0.0, all_values)
        assert result is True  # ASHA prunes the worst trial

    def test_top_quality_trial_not_pruned_by_asha_fallback(self) -> None:
        """Unreachable OrcaMind → fallback to ASHAPruner which keeps the best trial."""
        bad_client = MagicMock(spec=OrcaMindClient)
        bad_client.predict_performance = AsyncMock(side_effect=Exception("refused"))

        asha = ASHAPruner(min_resource=1, max_resource=81, reduction_factor=3)
        pruner = MetaPruner(bad_client, asha, min_steps_before_prediction=1)

        n = 9
        all_values = {f"t{i}": [float(i) / n] for i in range(n)}
        result = pruner.should_prune("t8", 1, float(n - 1) / n, all_values)
        assert result is False  # ASHA keeps the best trial

    def test_exception_does_not_propagate(self) -> None:
        """Any OrcaMind exception must be swallowed; should_prune must not raise."""
        bad_client = MagicMock(spec=OrcaMindClient)
        bad_client.predict_performance = AsyncMock(
            side_effect=RuntimeError("unexpected")
        )
        asha = ASHAPruner(min_resource=1, max_resource=81, reduction_factor=3)
        pruner = MetaPruner(bad_client, asha, min_steps_before_prediction=1)
        pruner.should_prune("t0", 5, 0.5, {})  # must not raise


# ---------------------------------------------------------------------------
# TestMetaPrunerEmptyMetrics
# ---------------------------------------------------------------------------


class TestMetaPrunerEmptyMetrics:
    def test_empty_final_metrics_falls_back_to_base_pruner(self) -> None:
        """When OrcaMind returns metrics with an empty final_metrics dict,
        _query_orcamind returns None and base_pruner makes the decision."""
        client = MagicMock(spec=OrcaMindClient)
        empty_metrics = PerformanceMetrics(
            experiment_id=uuid4(),
            final_metrics={},
        )
        client.predict_performance = AsyncMock(return_value=empty_metrics)

        base = MagicMock(spec=Pruner)
        base.should_prune = MagicMock(return_value=True)

        pruner = MetaPruner(client, base, prediction_threshold=0.3, min_steps_before_prediction=1)
        result = pruner.should_prune("t0", 5, 0.5, {})
        assert result is True
        base.should_prune.assert_called_once()

    def test_empty_final_metrics_base_says_no_prune(self) -> None:
        """With empty final_metrics and a permissive base_pruner, trial is not pruned."""
        client = MagicMock(spec=OrcaMindClient)
        empty_metrics = PerformanceMetrics(
            experiment_id=uuid4(),
            final_metrics={},
        )
        client.predict_performance = AsyncMock(return_value=empty_metrics)

        base = MagicMock(spec=Pruner)
        base.should_prune = MagicMock(return_value=False)

        pruner = MetaPruner(client, base, prediction_threshold=0.3, min_steps_before_prediction=1)
        result = pruner.should_prune("t0", 5, 0.5, {})
        assert result is False


# ---------------------------------------------------------------------------
# TestMetaPrunerThresholdEdgeCases
# ---------------------------------------------------------------------------


class TestMetaPrunerThresholdEdgeCases:
    def test_threshold_zero_only_prunes_negative_predictions(self) -> None:
        """With prediction_threshold=0.0, only a prediction < 0.0 triggers OrcaMind pruning."""
        client = MagicMock(spec=OrcaMindClient)
        # prediction=0.0 is not < 0.0, so OrcaMind does NOT prune; base decides
        client.predict_performance = AsyncMock(return_value=_make_metrics(0.0))

        base = MagicMock(spec=Pruner)
        base.should_prune = MagicMock(return_value=False)

        pruner = MetaPruner(client, base, prediction_threshold=0.0, min_steps_before_prediction=1)
        result = pruner.should_prune("t0", 5, 0.5, {})
        assert result is False
        base.should_prune.assert_called_once()

    def test_threshold_one_prunes_any_sub_one_prediction(self) -> None:
        """With prediction_threshold=1.0, any prediction < 1.0 triggers OrcaMind pruning."""
        client = MagicMock(spec=OrcaMindClient)
        client.predict_performance = AsyncMock(return_value=_make_metrics(0.99))

        base = MagicMock(spec=Pruner)
        base.should_prune = MagicMock(return_value=False)

        pruner = MetaPruner(client, base, prediction_threshold=1.0, min_steps_before_prediction=1)
        result = pruner.should_prune("t0", 5, 0.5, {})
        assert result is True
        base.should_prune.assert_not_called()

    def test_multiple_metrics_keys_max_is_used(self) -> None:
        """When final_metrics has multiple keys, the max value is used for comparison."""
        client = MagicMock(spec=OrcaMindClient)
        multi_metrics = PerformanceMetrics(
            experiment_id=uuid4(),
            final_metrics={"loss": 0.05, "accuracy": 0.95, "f1": 0.88},
        )
        client.predict_performance = AsyncMock(return_value=multi_metrics)

        base = MagicMock(spec=Pruner)
        base.should_prune = MagicMock(return_value=False)

        # max of {0.05, 0.95, 0.88} = 0.95; threshold=0.3 → 0.95 >= 0.3 → no OrcaMind prune
        pruner = MetaPruner(client, base, prediction_threshold=0.3, min_steps_before_prediction=1)
        result = pruner.should_prune("t0", 5, 0.5, {})
        assert result is False

    def test_multiple_metrics_low_max_triggers_prune(self) -> None:
        """max of final_metrics below threshold triggers OrcaMind pruning."""
        client = MagicMock(spec=OrcaMindClient)
        multi_metrics = PerformanceMetrics(
            experiment_id=uuid4(),
            final_metrics={"loss": 0.01, "accuracy": 0.02, "f1": 0.03},
        )
        client.predict_performance = AsyncMock(return_value=multi_metrics)

        base = MagicMock(spec=Pruner)

        # max = 0.03 < 0.3 → pruned by OrcaMind
        pruner = MetaPruner(client, base, prediction_threshold=0.3, min_steps_before_prediction=1)
        result = pruner.should_prune("t0", 5, 0.5, {})
        assert result is True


# ---------------------------------------------------------------------------
# TestMetaPrunerCurveEmbedding
# ---------------------------------------------------------------------------


class TestMetaPrunerCurveEmbedding:
    def test_task_embedding_includes_current_value(self, mock_client: MagicMock) -> None:
        """The task embedding sent to OrcaMind must include current_value appended to history."""
        pruner = MetaPruner(mock_client, MagicMock(spec=Pruner), min_steps_before_prediction=1)
        pruner._base_pruner.should_prune = MagicMock(return_value=False)

        all_values = {"t0": [0.1, 0.2, 0.3]}
        pruner.should_prune("t0", 3, 0.4, all_values)

        call_kwargs = mock_client.predict_performance.call_args
        embedding = call_kwargs.kwargs.get(
            "task_embedding", call_kwargs.args[0] if call_kwargs.args else None
        )
        assert embedding is not None
        assert embedding[-1] == pytest.approx(0.4)  # current_value is last element

    def test_task_embedding_trial_not_in_all_trial_values(self, mock_client: MagicMock) -> None:
        """When trial_id is absent from all_trial_values, curve starts from [current_value]."""
        pruner = MetaPruner(mock_client, MagicMock(spec=Pruner), min_steps_before_prediction=1)
        pruner._base_pruner.should_prune = MagicMock(return_value=False)

        # trial_id "t_new" not present in all_trial_values
        pruner.should_prune("t_new", 5, 0.7, {"t_other": [0.5, 0.6]})

        call_kwargs = mock_client.predict_performance.call_args
        embedding = call_kwargs.kwargs.get(
            "task_embedding", call_kwargs.args[0] if call_kwargs.args else None
        )
        assert embedding is not None
        assert len(embedding) == 1  # [] + [0.7]
        assert embedding[0] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# TestMetaPrunerMedianBasePrunerIntegration
# ---------------------------------------------------------------------------


class TestMetaPrunerMedianBasePrunerIntegration:
    def test_meta_pruner_with_median_base_falls_back_correctly(self) -> None:
        """Integration: OrcaMind fails → MetaPruner delegates to MedianStoppingPruner."""
        from orcalab.pruning.median import MedianStoppingPruner

        bad_client = MagicMock(spec=OrcaMindClient)
        bad_client.predict_performance = AsyncMock(side_effect=ConnectionError("down"))

        median_base = MedianStoppingPruner(warmup_steps=1)
        pruner = MetaPruner(bad_client, median_base, min_steps_before_prediction=1)

        # Two peers: best = 0.9; current trial value = 0.1 → pruned by median
        all_values = {"t1": [0.9], "t2": [0.8]}
        result = pruner.should_prune("t0", 1, 0.1, all_values)
        assert result is True  # median fallback prunes the underperformer

    def test_meta_pruner_with_median_base_keeps_good_trial(self) -> None:
        """Integration: OrcaMind fails → MetaPruner delegates to MedianStoppingPruner
        which keeps a trial performing above the median."""
        from orcalab.pruning.median import MedianStoppingPruner

        bad_client = MagicMock(spec=OrcaMindClient)
        bad_client.predict_performance = AsyncMock(side_effect=ConnectionError("down"))

        median_base = MedianStoppingPruner(warmup_steps=1)
        pruner = MetaPruner(bad_client, median_base, min_steps_before_prediction=1)

        # Two peers: best = 0.3, 0.5; median = 0.4; current = 0.9 → not pruned
        all_values = {"t1": [0.3], "t2": [0.5]}
        result = pruner.should_prune("t0", 1, 0.9, all_values)
        assert result is False
