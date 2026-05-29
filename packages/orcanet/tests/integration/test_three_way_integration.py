"""Integration tests for the three-way TransferPipeline (OrcaNet ↔ OrcaMind ↔ OrcaLab).

All external dependencies are mocked so no running services are required.
Tests verify the pipeline's coordination logic, error propagation, and
database-persistence contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import httpx
import pytest

from orca_shared.schemas.task import Task
from orca_shared.schemas.training import ExperimentResult
from orca_shared.schemas.transfer import TransferMapping
from orcanet.integration.pipeline import (
    ServiceUnavailableError,
    TransferPipeline,
    TransferValidationResult,
)
from orcanet.transfer.types import TransferScore


# ---------------------------------------------------------------------------
# Shared test data helpers
# ---------------------------------------------------------------------------

_NOW = datetime.now(timezone.utc)


def _make_task(
    task_id: UUID | None = None,
    name: str = "task",
    domain: str = "vision",
    task_type: str = "classification",
) -> Task:
    """Return a minimal :class:`Task` instance for use in tests."""
    tid = task_id or uuid4()
    return Task(
        task_id=tid,
        name=name,
        domain=domain,
        task_type=task_type,
        n_samples=1000,
        n_features=25,
        n_classes=10,
        dataset_uri=None,
        metadata=None,
        embedding_id=None,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _make_mapping(source_id: UUID, target_id: UUID, score: float = 0.75) -> TransferMapping:
    """Return a minimal :class:`TransferMapping` instance for use in tests."""
    return TransferMapping(
        mapping_id=uuid4(),
        source_task_id=source_id,
        target_task_id=target_id,
        transfer_score=score,
        transfer_type="feature",
        metadata=None,
        created_at=_NOW,
    )


def _make_experiment_result(
    status: str = "COMPLETED",
    accuracy: float = 0.88,
    baseline: float = 0.76,
) -> ExperimentResult:
    """Return a minimal :class:`ExperimentResult` instance for use in tests."""
    return ExperimentResult(
        experiment_id=uuid4(),
        task_id=None,
        model_id=None,
        status=status,
        mlflow_run_id=None,
        started_at=_NOW,
        completed_at=_NOW,
        metrics={"accuracy": accuracy, "baseline_accuracy": baseline},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def source_id() -> UUID:
    """Provide a random UUID for the source task."""
    return uuid4()


@pytest.fixture
def target_id() -> UUID:
    """Provide a random UUID for the target task."""
    return uuid4()


@pytest.fixture
def source_task(source_id: UUID) -> Task:
    """Return a vision-domain source task keyed to *source_id*."""
    return _make_task(task_id=source_id, name="source-task")


@pytest.fixture
def target_task(target_id: UUID) -> Task:
    """Return an NLP-domain target task keyed to *target_id*."""
    return _make_task(task_id=target_id, name="target-task", domain="nlp")


@pytest.fixture
def high_score() -> TransferScore:
    """Return a :class:`TransferScore` with overall 0.82 (above the 0.4 threshold)."""
    return TransferScore(
        overall=0.82,
        layer_scores={"layer0": 0.9, "layer1": 0.74},
        recommended_layers=["layer0", "layer1"],
        reasoning="Strong CKA alignment",
    )


@pytest.fixture
def low_score() -> TransferScore:
    """Return a :class:`TransferScore` with overall 0.25 (below the 0.4 threshold)."""
    return TransferScore(
        overall=0.25,
        layer_scores={"layer0": 0.2},
        recommended_layers=[],
        reasoning="Weak domain overlap",
    )


@pytest.fixture
def mock_strategy(high_score: TransferScore) -> MagicMock:
    """Return a mock transfer strategy that always returns *high_score*."""
    strategy = MagicMock()
    strategy.score_transfer = MagicMock(return_value=high_score)
    return strategy


@pytest.fixture
def mock_orcamind() -> AsyncMock:
    """Return a mock OrcaMindClient whose get_best_model returns a consistent ModelSummary."""
    client = AsyncMock()
    model_summary = MagicMock()
    _model_id = uuid4()
    model_summary.model_id = _model_id
    model_summary.name = "resnet18"
    model_summary.architecture = "resnet"
    model_summary.model_dump = MagicMock(
        return_value={"model_id": str(_model_id), "name": "resnet18", "architecture": "resnet"}
    )
    client.get_best_model = AsyncMock(return_value=model_summary)
    return client


@pytest.fixture
def mock_orcalab(target_id: UUID) -> AsyncMock:
    """Return a mock OrcaLabClient that successfully creates and completes an experiment."""
    client = AsyncMock()
    exp_id = str(uuid4())
    client.create_experiment = AsyncMock(return_value=exp_id)
    client.wait_for_completion = AsyncMock(return_value=_make_experiment_result())
    return client


@pytest.fixture
def mock_repo(source_task: Task, target_task: Task, source_id: UUID, target_id: UUID) -> AsyncMock:
    """Return a mock TaskRepository with source/target tasks pre-seeded."""
    repo = AsyncMock()

    async def _get_by_id(task_id: UUID) -> Task | None:
        if task_id == source_id:
            return source_task
        if task_id == target_id:
            return target_task
        return None

    repo.get_by_id.side_effect = _get_by_id
    repo.save_transfer_mapping = AsyncMock(
        return_value=_make_mapping(source_id, target_id)
    )
    return repo


@pytest.fixture
def pipeline(
    mock_orcamind: AsyncMock,
    mock_orcalab: AsyncMock,
    mock_strategy: MagicMock,
    mock_repo: AsyncMock,
) -> TransferPipeline:
    """Return a :class:`TransferPipeline` wired to mock upstream clients."""
    return TransferPipeline(
        orcamind_client=mock_orcamind,
        orcalab_client=mock_orcalab,
        transfer_strategies={"feature": mock_strategy},
        task_repository=mock_repo,
    )


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestFullPipelineHappyPath:
    """Verify nominal pipeline behaviour when all upstreams respond successfully."""

    @pytest.mark.asyncio
    async def test_returns_transfer_validation_result(
        self,
        pipeline: TransferPipeline,
        source_id: UUID,
        target_id: UUID,
    ) -> None:
        """Pipeline result is an instance of TransferValidationResult."""
        result = await pipeline.recommend_and_validate(
            str(source_id), str(target_id), validate=True
        )
        assert isinstance(result, TransferValidationResult)

    @pytest.mark.asyncio
    async def test_score_is_non_negative(
        self,
        pipeline: TransferPipeline,
        source_id: UUID,
        target_id: UUID,
    ) -> None:
        """The overall transfer score is non-negative."""
        result = await pipeline.recommend_and_validate(str(source_id), str(target_id))
        assert result.score.overall >= 0.0

    @pytest.mark.asyncio
    async def test_experiment_result_present_when_score_above_threshold(
        self,
        pipeline: TransferPipeline,
        source_id: UUID,
        target_id: UUID,
    ) -> None:
        """Experiment result is populated when the score exceeds the 0.4 validation threshold."""
        result = await pipeline.recommend_and_validate(
            str(source_id), str(target_id), validate=True
        )
        # score fixture is 0.82 > 0.4, so validation should run
        assert result.experiment_result is not None

    @pytest.mark.asyncio
    async def test_mapping_source_id_matches_input(
        self,
        pipeline: TransferPipeline,
        source_id: UUID,
        target_id: UUID,
        mock_repo: AsyncMock,
    ) -> None:
        """save_transfer_mapping is called with the correct source and target UUIDs."""
        result = await pipeline.recommend_and_validate(str(source_id), str(target_id))
        saved_args = mock_repo.save_transfer_mapping.call_args
        assert saved_args.kwargs["source_task_id"] == source_id
        assert saved_args.kwargs["target_task_id"] == target_id

    @pytest.mark.asyncio
    async def test_improvement_computed_from_experiment_metrics(
        self,
        pipeline: TransferPipeline,
        source_id: UUID,
        target_id: UUID,
    ) -> None:
        """improvement_over_baseline equals accuracy minus baseline_accuracy."""
        result = await pipeline.recommend_and_validate(str(source_id), str(target_id))
        # accuracy=0.88, baseline=0.76 → improvement ≈ 0.12
        assert result.improvement_over_baseline is not None
        assert result.improvement_over_baseline == pytest.approx(0.12, abs=1e-6)

    @pytest.mark.asyncio
    async def test_orcamind_get_best_model_called_with_source_uuid(
        self,
        pipeline: TransferPipeline,
        mock_orcamind: AsyncMock,
        source_id: UUID,
        target_id: UUID,
    ) -> None:
        """OrcaMind's get_best_model is called exactly once with the source UUID."""
        await pipeline.recommend_and_validate(str(source_id), str(target_id))
        mock_orcamind.get_best_model.assert_awaited_once_with(source_id)

    @pytest.mark.asyncio
    async def test_orcalab_create_experiment_called_with_correct_tags(
        self,
        pipeline: TransferPipeline,
        mock_orcalab: AsyncMock,
        source_id: UUID,
        target_id: UUID,
    ) -> None:
        """OrcaLab's create_experiment receives the expected validation tags."""
        await pipeline.recommend_and_validate(str(source_id), str(target_id))
        call_kwargs = mock_orcalab.create_experiment.call_args.kwargs
        assert "transfer_validation" in call_kwargs["tags"]
        assert f"source:{source_id}" in call_kwargs["tags"]
        assert "strategy:feature" in call_kwargs["tags"]


# ---------------------------------------------------------------------------
# Validation skip when score is below threshold
# ---------------------------------------------------------------------------


class TestValidationSkippedBelowThreshold:
    """Verify that OrcaLab is not called when the score is below the threshold."""

    @pytest.mark.asyncio
    async def test_experiment_result_none_when_score_below_threshold(
        self,
        mock_orcamind: AsyncMock,
        mock_orcalab: AsyncMock,
        mock_repo: AsyncMock,
        source_id: UUID,
        target_id: UUID,
        low_score: TransferScore,
    ) -> None:
        """experiment_result is None and OrcaLab is not called when score < 0.4."""
        strategy = MagicMock()
        strategy.score_transfer = MagicMock(return_value=low_score)
        pl = TransferPipeline(
            orcamind_client=mock_orcamind,
            orcalab_client=mock_orcalab,
            transfer_strategies={"feature": strategy},
            task_repository=mock_repo,
        )
        result = await pl.recommend_and_validate(str(source_id), str(target_id), validate=True)
        assert result.experiment_result is None
        mock_orcalab.create_experiment.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_experiment_result_none_when_validate_false(
        self,
        pipeline: TransferPipeline,
        mock_orcalab: AsyncMock,
        source_id: UUID,
        target_id: UUID,
    ) -> None:
        """experiment_result is None and OrcaLab is not called when validate=False."""
        result = await pipeline.recommend_and_validate(
            str(source_id), str(target_id), validate=False
        )
        assert result.experiment_result is None
        mock_orcalab.create_experiment.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mapping_saved_even_when_validation_skipped(
        self,
        mock_orcamind: AsyncMock,
        mock_orcalab: AsyncMock,
        mock_repo: AsyncMock,
        source_id: UUID,
        target_id: UUID,
        low_score: TransferScore,
    ) -> None:
        """The transfer mapping is persisted even when OrcaLab validation is skipped."""
        strategy = MagicMock()
        strategy.score_transfer = MagicMock(return_value=low_score)
        pl = TransferPipeline(
            orcamind_client=mock_orcamind,
            orcalab_client=mock_orcalab,
            transfer_strategies={"feature": strategy},
            task_repository=mock_repo,
        )
        await pl.recommend_and_validate(str(source_id), str(target_id), validate=True)
        mock_repo.save_transfer_mapping.assert_awaited_once()


# ---------------------------------------------------------------------------
# OrcaLab timeout scenario
# ---------------------------------------------------------------------------


class TestOrcaLabTimeout:
    """Verify graceful handling of OrcaLab timeouts."""

    @pytest.mark.asyncio
    async def test_timeout_propagated_as_timeout_error(
        self,
        mock_orcamind: AsyncMock,
        mock_orcalab: AsyncMock,
        mock_repo: AsyncMock,
        mock_strategy: MagicMock,
        source_id: UUID,
        target_id: UUID,
    ) -> None:
        """Pipeline completes without raising when OrcaLab times out; experiment_result is None."""
        mock_orcalab.wait_for_completion = AsyncMock(side_effect=TimeoutError("timed out"))
        pl = TransferPipeline(
            orcamind_client=mock_orcamind,
            orcalab_client=mock_orcalab,
            transfer_strategies={"feature": mock_strategy},
            task_repository=mock_repo,
        )
        result = await pl.recommend_and_validate(str(source_id), str(target_id), validate=True)
        # Timeout is caught; pipeline still completes and mapping is saved
        assert result.experiment_result is None
        mock_repo.save_transfer_mapping.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mapping_metadata_experiment_id_none_on_timeout(
        self,
        mock_orcamind: AsyncMock,
        mock_orcalab: AsyncMock,
        mock_repo: AsyncMock,
        mock_strategy: MagicMock,
        source_id: UUID,
        target_id: UUID,
    ) -> None:
        """validated_performance in the mapping metadata is None after a timeout."""
        # create_experiment succeeds, wait_for_completion times out
        mock_orcalab.wait_for_completion = AsyncMock(side_effect=TimeoutError("timed out"))
        pl = TransferPipeline(
            orcamind_client=mock_orcamind,
            orcalab_client=mock_orcalab,
            transfer_strategies={"feature": mock_strategy},
            task_repository=mock_repo,
        )
        await pl.recommend_and_validate(str(source_id), str(target_id), validate=True)
        saved_meta = mock_repo.save_transfer_mapping.call_args.kwargs["metadata"]
        assert saved_meta["validated_performance"] is None


# ---------------------------------------------------------------------------
# OrcaMind unavailability
# ---------------------------------------------------------------------------


class TestOrcaMindUnavailable:
    """Verify that OrcaMind connectivity failures surface as ServiceUnavailableError."""

    @pytest.mark.asyncio
    async def test_connect_error_raises_service_unavailable(
        self,
        mock_orcalab: AsyncMock,
        mock_strategy: MagicMock,
        mock_repo: AsyncMock,
        source_id: UUID,
        target_id: UUID,
    ) -> None:
        """httpx.ConnectError from OrcaMind is wrapped in ServiceUnavailableError."""
        orcamind = AsyncMock()
        orcamind.get_best_model = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        pl = TransferPipeline(
            orcamind_client=orcamind,
            orcalab_client=mock_orcalab,
            transfer_strategies={"feature": mock_strategy},
            task_repository=mock_repo,
        )
        with pytest.raises(ServiceUnavailableError, match="OrcaMind is unreachable"):
            await pl.recommend_and_validate(str(source_id), str(target_id))

    @pytest.mark.asyncio
    async def test_5xx_from_orcamind_raises_service_unavailable(
        self,
        mock_orcalab: AsyncMock,
        mock_strategy: MagicMock,
        mock_repo: AsyncMock,
        source_id: UUID,
        target_id: UUID,
    ) -> None:
        """A 5xx HTTP response from OrcaMind is wrapped in ServiceUnavailableError."""
        orcamind = AsyncMock()
        resp = MagicMock()
        resp.status_code = 503
        orcamind.get_best_model = AsyncMock(
            side_effect=httpx.HTTPStatusError("", request=MagicMock(), response=resp)
        )
        pl = TransferPipeline(
            orcamind_client=orcamind,
            orcalab_client=mock_orcalab,
            transfer_strategies={"feature": mock_strategy},
            task_repository=mock_repo,
        )
        with pytest.raises(ServiceUnavailableError):
            await pl.recommend_and_validate(str(source_id), str(target_id))


# ---------------------------------------------------------------------------
# Task not found
# ---------------------------------------------------------------------------


class TestTaskNotFound:
    """Verify that missing source or target tasks raise ValueError."""

    @pytest.mark.asyncio
    async def test_missing_source_task_raises_value_error(
        self,
        pipeline: TransferPipeline,
        target_id: UUID,
    ) -> None:
        """ValueError is raised when the source task UUID is not in the repository."""
        with pytest.raises(ValueError, match="Source task"):
            await pipeline.recommend_and_validate(str(uuid4()), str(target_id))

    @pytest.mark.asyncio
    async def test_missing_target_task_raises_value_error(
        self,
        pipeline: TransferPipeline,
        source_id: UUID,
    ) -> None:
        """ValueError is raised when the target task UUID is not in the repository."""
        with pytest.raises(ValueError, match="Target task"):
            await pipeline.recommend_and_validate(str(source_id), str(uuid4()))


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------


class TestStrategySelection:
    """Verify that the chosen strategy is correctly applied and persisted."""

    @pytest.mark.asyncio
    async def test_default_strategy_is_feature(
        self,
        mock_orcamind: AsyncMock,
        mock_orcalab: AsyncMock,
        mock_repo: AsyncMock,
        mock_strategy: MagicMock,
        source_id: UUID,
        target_id: UUID,
    ) -> None:
        """The default strategy_name is 'feature', reflected in the persisted mapping."""
        pl = TransferPipeline(
            orcamind_client=mock_orcamind,
            orcalab_client=mock_orcalab,
            transfer_strategies={"feature": mock_strategy},
            task_repository=mock_repo,
        )
        result = await pl.recommend_and_validate(str(source_id), str(target_id))
        saved_args = mock_repo.save_transfer_mapping.call_args.kwargs
        assert saved_args["transfer_type"] == "feature"

    @pytest.mark.asyncio
    async def test_unknown_strategy_raises_key_error(
        self,
        pipeline: TransferPipeline,
        source_id: UUID,
        target_id: UUID,
    ) -> None:
        """KeyError is raised when the requested strategy is not registered."""
        with pytest.raises(KeyError):
            await pipeline.recommend_and_validate(
                str(source_id), str(target_id), strategy_name="nonexistent"
            )
