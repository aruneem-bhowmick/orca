"""Validate all orca_shared Pydantic v2 schemas with valid and invalid data."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orca_shared.schemas.embedding import Embedding, SimilarityResult
from orca_shared.schemas.metrics import MetricPoint, PerformanceMetrics
from orca_shared.schemas.model import ModelConfig, ModelSummary
from orca_shared.schemas.recommendation import FeedbackRequest, ModelRecommendation, RecommendationRequest
from orca_shared.schemas.task import DatasetSummary, Task, TaskCreate, TaskSummary
from orca_shared.schemas.training import ExperimentResult, TrainingConfig
from orca_shared.schemas.transfer import TransferMapping, TransferRecommendation, TransferScore

NOW = datetime.now(timezone.utc)
TASK_ID = uuid.uuid4()
MODEL_ID = uuid.uuid4()
EXPERIMENT_ID = uuid.uuid4()
EMBEDDING_ID = uuid.uuid4()
MAPPING_ID = uuid.uuid4()


# ---------------------------------------------------------------------------
# task schemas
# ---------------------------------------------------------------------------


class TestTaskCreate:
    def test_valid(self):
        tc = TaskCreate(name="iris", task_type="classification")
        assert tc.name == "iris"
        assert tc.task_type == "classification"
        assert tc.domain is None

    def test_missing_required_name(self):
        with pytest.raises(ValidationError):
            TaskCreate(task_type="classification")  # type: ignore[call-arg]

    def test_missing_required_task_type(self):
        with pytest.raises(ValidationError):
            TaskCreate(name="iris")  # type: ignore[call-arg]


class TestTask:
    def test_valid(self):
        t = Task(
            task_id=TASK_ID,
            name="iris",
            task_type="classification",
            created_at=NOW,
            updated_at=NOW,
        )
        assert t.task_id == TASK_ID

    def test_missing_task_id(self):
        with pytest.raises(ValidationError):
            Task(name="iris", task_type="classification", created_at=NOW, updated_at=NOW)  # type: ignore[call-arg]


class TestTaskSummary:
    def test_valid(self):
        ts = TaskSummary(task_id=TASK_ID, name="iris", task_type="classification")
        assert ts.name == "iris"

    def test_missing_task_id(self):
        with pytest.raises(ValidationError):
            TaskSummary(name="iris", task_type="classification")  # type: ignore[call-arg]


class TestDatasetSummary:
    def test_valid(self):
        ds = DatasetSummary(dataset_uri="s3://bucket/data.parquet", n_samples=150, n_features=4)
        assert ds.n_samples == 150

    def test_missing_uri(self):
        with pytest.raises(ValidationError):
            DatasetSummary(n_samples=150, n_features=4)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# model schemas
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_valid(self):
        mc = ModelConfig(model_id=MODEL_ID, name="mlp", config={"hidden": 128})
        assert mc.config == {"hidden": 128}

    def test_missing_config(self):
        with pytest.raises(ValidationError):
            ModelConfig(model_id=MODEL_ID, name="mlp")  # type: ignore[call-arg]


class TestModelSummary:
    def test_valid(self):
        ms = ModelSummary(model_id=MODEL_ID, name="mlp")
        assert ms.model_id == MODEL_ID

    def test_missing_model_id(self):
        with pytest.raises(ValidationError):
            ModelSummary(name="mlp")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# training schemas
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    def test_defaults(self):
        tc = TrainingConfig()
        assert tc.batch_size == 32
        assert tc.lr == pytest.approx(1e-3)
        assert tc.epochs == 10

    def test_custom(self):
        tc = TrainingConfig(batch_size=64, lr=5e-4, epochs=20, optimizer="sgd")
        assert tc.batch_size == 64


class TestExperimentResult:
    def test_valid(self):
        er = ExperimentResult(experiment_id=EXPERIMENT_ID, status="pending")
        assert er.status == "pending"

    def test_missing_experiment_id(self):
        with pytest.raises(ValidationError):
            ExperimentResult(status="pending")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# metrics schemas
# ---------------------------------------------------------------------------


class TestMetricPoint:
    def test_valid(self):
        mp = MetricPoint(name="accuracy", value=0.95)
        assert mp.is_final is False

    def test_missing_name(self):
        with pytest.raises(ValidationError):
            MetricPoint(value=0.95)  # type: ignore[call-arg]


class TestPerformanceMetrics:
    def test_valid(self):
        pm = PerformanceMetrics(
            experiment_id=EXPERIMENT_ID, final_metrics={"accuracy": 0.95}
        )
        assert pm.final_metrics["accuracy"] == pytest.approx(0.95)

    def test_missing_experiment_id(self):
        with pytest.raises(ValidationError):
            PerformanceMetrics(final_metrics={"accuracy": 0.95})  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# embedding schemas
# ---------------------------------------------------------------------------


class TestEmbedding:
    def test_valid(self):
        emb = Embedding(
            embedding_id=EMBEDDING_ID,
            embedding_vector=[0.1, 0.2, 0.3],
            dimension=3,
            created_at=NOW,
        )
        assert emb.dimension == 3

    def test_missing_vector(self):
        with pytest.raises(ValidationError):
            Embedding(embedding_id=EMBEDDING_ID, dimension=3, created_at=NOW)  # type: ignore[call-arg]


class TestSimilarityResult:
    def test_valid(self):
        sr = SimilarityResult(task_id=TASK_ID, score=0.9, rank=1)
        assert sr.rank == 1

    def test_missing_score(self):
        with pytest.raises(ValidationError):
            SimilarityResult(task_id=TASK_ID, rank=1)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# transfer schemas
# ---------------------------------------------------------------------------


class TestTransferMapping:
    def test_valid(self):
        src = uuid.uuid4()
        tgt = uuid.uuid4()
        tm = TransferMapping(
            mapping_id=MAPPING_ID,
            source_task_id=src,
            target_task_id=tgt,
            transfer_score=0.8,
            created_at=NOW,
        )
        assert tm.transfer_score == pytest.approx(0.8)

    def test_missing_source(self):
        with pytest.raises(ValidationError):
            TransferMapping(
                mapping_id=MAPPING_ID,
                target_task_id=uuid.uuid4(),
                transfer_score=0.8,
                created_at=NOW,
            )  # type: ignore[call-arg]


class TestTransferScore:
    def test_valid(self):
        ts = TransferScore(source_task_id=TASK_ID, target_task_id=uuid.uuid4(), score=0.7)
        assert ts.score == pytest.approx(0.7)


class TestTransferRecommendation:
    def test_valid(self):
        src = uuid.uuid4()
        tgt = uuid.uuid4()
        mapping = TransferMapping(
            mapping_id=MAPPING_ID,
            source_task_id=src,
            target_task_id=tgt,
            transfer_score=0.9,
            created_at=NOW,
        )
        tr = TransferRecommendation(
            target_task_id=tgt, recommended_sources=[mapping], top_score=0.9
        )
        assert len(tr.recommended_sources) == 1


# ---------------------------------------------------------------------------
# recommendation schemas
# ---------------------------------------------------------------------------


class TestRecommendationRequest:
    def test_valid(self):
        rr = RecommendationRequest(task_embedding=[0.1, 0.2])
        assert rr.top_k == 5

    def test_missing_embedding(self):
        with pytest.raises(ValidationError):
            RecommendationRequest()  # type: ignore[call-arg]


class TestModelRecommendation:
    def test_valid(self):
        mr = ModelRecommendation(
            task_id=TASK_ID, model_id=MODEL_ID, predicted_score=0.88
        )
        assert mr.predicted_score == pytest.approx(0.88)

    def test_missing_predicted_score(self):
        with pytest.raises(ValidationError):
            ModelRecommendation(task_id=TASK_ID, model_id=MODEL_ID)  # type: ignore[call-arg]


class TestFeedbackRequest:
    def test_valid(self):
        fr = FeedbackRequest(
            experiment_id=EXPERIMENT_ID, actual_metric=0.91, metric_name="f1"
        )
        assert fr.metric_name == "f1"

    def test_missing_metric_name(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(experiment_id=EXPERIMENT_ID, actual_metric=0.91)  # type: ignore[call-arg]
