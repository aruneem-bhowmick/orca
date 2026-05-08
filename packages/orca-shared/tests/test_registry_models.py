"""Tests for ORM model structure, column mapping, relationships, and session utilities."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from orca_shared.registry.models import (
    Base,
    Embedding,
    Experiment,
    Model,
    Performance,
    SearchSpace,
    Task,
    TransferMapping,
    get_engine,
    get_session,
)


# ---------------------------------------------------------------------------
# Table names
# ---------------------------------------------------------------------------


class TestTableNames:
    def test_tasks(self):
        assert Task.__tablename__ == "tasks"

    def test_embeddings(self):
        assert Embedding.__tablename__ == "embeddings"

    def test_models(self):
        assert Model.__tablename__ == "models"

    def test_experiments(self):
        assert Experiment.__tablename__ == "experiments"

    def test_performances(self):
        assert Performance.__tablename__ == "performances"

    def test_transfer_mappings(self):
        assert TransferMapping.__tablename__ == "transfer_mappings"

    def test_search_spaces(self):
        assert SearchSpace.__tablename__ == "search_spaces"


# ---------------------------------------------------------------------------
# Column presence and DB-level names
# ---------------------------------------------------------------------------


class TestColumnMapping:
    def test_task_has_all_spec_columns(self):
        # column_attrs is keyed by Python attribute name; .columns uses DB column name
        col_names = {prop.key for prop in inspect(Task).column_attrs}
        required = {
            "task_id", "name", "domain", "task_type", "n_samples", "n_features",
            "n_classes", "dataset_uri", "task_metadata", "embedding_id",
            "created_at", "updated_at",
        }
        assert required.issubset(col_names)

    def test_task_metadata_python_attr_maps_to_metadata_db_column(self):
        prop = inspect(Task).attrs["task_metadata"]
        assert prop.columns[0].name == "metadata"

    def test_performance_metadata_python_attr_maps_to_metadata_db_column(self):
        prop = inspect(Performance).attrs["perf_metadata"]
        assert prop.columns[0].name == "metadata"

    def test_transfer_mapping_metadata_python_attr_maps_to_metadata_db_column(self):
        prop = inspect(TransferMapping).attrs["mapping_metadata"]
        assert prop.columns[0].name == "metadata"

    def test_embedding_has_all_spec_columns(self):
        col_names = {prop.key for prop in inspect(Embedding).column_attrs}
        required = {
            "embedding_id", "task_id", "embedding_type", "embedding_vector",
            "dimension", "model_version", "created_at",
        }
        assert required.issubset(col_names)

    def test_experiment_has_all_spec_columns(self):
        col_names = {prop.key for prop in inspect(Experiment).column_attrs}
        required = {
            "experiment_id", "task_id", "model_id", "training_config",
            "status", "mlflow_run_id", "started_at", "completed_at", "created_by",
        }
        assert required.issubset(col_names)

    def test_search_space_has_parent_id_for_self_reference(self):
        col_names = {prop.key for prop in inspect(SearchSpace).column_attrs}
        assert "parent_id" in col_names

    def test_performance_is_final_defaults_false(self):
        col = inspect(Performance).attrs["is_final"].columns[0]
        assert col.default.arg is False

    def test_task_primary_key_is_task_id(self):
        pk_cols = [c.key for c in inspect(Task).primary_key]
        assert pk_cols == ["task_id"]

    def test_embedding_primary_key_is_embedding_id(self):
        pk_cols = [c.key for c in inspect(Embedding).primary_key]
        assert pk_cols == ["embedding_id"]

    def test_search_space_primary_key_is_search_space_id(self):
        pk_cols = [c.key for c in inspect(SearchSpace).primary_key]
        assert pk_cols == ["search_space_id"]


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------


class TestRelationships:
    def _rel_names(self, model):
        return {r.key for r in inspect(model).relationships}

    def test_task_has_experiments(self):
        assert "experiments" in self._rel_names(Task)

    def test_task_has_embeddings(self):
        assert "embeddings" in self._rel_names(Task)

    def test_task_has_current_embedding(self):
        assert "current_embedding" in self._rel_names(Task)

    def test_task_has_source_and_target_mappings(self):
        names = self._rel_names(Task)
        assert "source_mappings" in names
        assert "target_mappings" in names

    def test_experiment_has_task_model_and_performances(self):
        names = self._rel_names(Experiment)
        assert {"task", "model", "performances"}.issubset(names)

    def test_performance_back_references_experiment(self):
        assert "experiment" in self._rel_names(Performance)

    def test_embedding_back_references_task(self):
        assert "task" in self._rel_names(Embedding)

    def test_search_space_self_referential_parent_and_children(self):
        names = self._rel_names(SearchSpace)
        assert "parent" in names
        assert "children" in names

    def test_transfer_mapping_has_source_and_target_task(self):
        names = self._rel_names(TransferMapping)
        assert "source_task" in names
        assert "target_task" in names

    def test_model_has_experiments(self):
        assert "experiments" in self._rel_names(Model)


# ---------------------------------------------------------------------------
# All models share the same declarative base
# ---------------------------------------------------------------------------


class TestBase:
    def test_all_subclass_base(self):
        for cls in (Task, Embedding, Model, Experiment, Performance, TransferMapping, SearchSpace):
            assert issubclass(cls, Base), f"{cls.__name__} does not subclass Base"

    def test_shared_metadata_object(self):
        assert Task.metadata is Embedding.metadata is Model.metadata


# ---------------------------------------------------------------------------
# get_engine
# ---------------------------------------------------------------------------


class TestGetEngine:
    def test_normalises_postgresql_scheme(self):
        engine = get_engine("postgresql://user:pass@localhost:5432/db")
        assert "postgresql+asyncpg" in str(engine.url)

    def test_asyncpg_url_not_double_replaced(self):
        engine = get_engine("postgresql+asyncpg://user:pass@localhost:5432/db")
        assert str(engine.url).count("asyncpg") == 1

    def test_returns_async_engine(self):
        engine = get_engine("postgresql://user:pass@localhost:5432/db")
        assert isinstance(engine, AsyncEngine)

    def test_echo_disabled(self):
        engine = get_engine("postgresql://user:pass@localhost:5432/db")
        assert engine.echo is False

    def test_preserves_database_name(self):
        engine = get_engine("postgresql://user:pass@localhost:5432/orca_registry")
        assert "orca_registry" in str(engine.url)


# ---------------------------------------------------------------------------
# get_session
# ---------------------------------------------------------------------------


class TestGetSession:
    @pytest.mark.asyncio
    async def test_yields_session_from_factory(self):
        """get_session must yield an AsyncSession obtained from async_sessionmaker."""
        sentinel = object()
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_begin = AsyncMock()
        mock_begin.__aenter__ = AsyncMock(return_value=None)
        mock_begin.__aexit__ = AsyncMock(return_value=False)
        mock_session.begin = MagicMock(return_value=mock_begin)

        mock_factory = MagicMock(return_value=mock_session)

        with patch("orca_shared.registry.models.async_sessionmaker", return_value=mock_factory):
            mock_engine = MagicMock()
            async with get_session(mock_engine) as session:
                assert session is mock_session

    @pytest.mark.asyncio
    async def test_opens_transaction(self):
        """get_session must enter a begin() transaction block."""
        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_begin = AsyncMock()
        mock_begin.__aenter__ = AsyncMock(return_value=None)
        mock_begin.__aexit__ = AsyncMock(return_value=False)
        mock_session.begin = MagicMock(return_value=mock_begin)

        mock_factory = MagicMock(return_value=mock_session)

        with patch("orca_shared.registry.models.async_sessionmaker", return_value=mock_factory):
            async with get_session(MagicMock()):
                pass

        mock_session.begin.assert_called_once()


# ---------------------------------------------------------------------------
# ORM instantiation without a database
# ---------------------------------------------------------------------------


class TestOrmInstantiation:
    def test_task_instantiates_with_required_fields(self):
        import uuid
        t = Task(task_id=uuid.uuid4(), name="iris", task_type="classification")
        assert t.name == "iris"

    def test_task_metadata_attr_accessible(self):
        t = Task(name="x", task_type="clf")
        assert hasattr(t, "task_metadata")
        assert t.task_metadata is None

    def test_performance_is_final_default_false(self):
        import uuid
        p = Performance(performance_id=uuid.uuid4(), is_final=False)
        assert p.is_final is False

    def test_search_space_instantiates(self):
        import uuid
        ss = SearchSpace(search_space_id=uuid.uuid4(), definition={"lr": [0.001, 0.01]})
        assert ss.definition == {"lr": [0.001, 0.01]}
