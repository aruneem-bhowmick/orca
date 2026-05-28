"""Unit tests for orcanet.reasoning.tools."""

from __future__ import annotations

import importlib
import json

import pytest
import torch


def _mod(name: str):
    """Return the real module object, bypassing package attribute shadowing."""
    return importlib.import_module(f"orcanet.reasoning.tools.{name}")


@pytest.fixture(autouse=True)
def _reset_reasoning_tool_state():
    """Reset all module-level tool registries before and after each test."""
    tr = _mod("task_retrieval_tool")
    es = _mod("embedding_similarity_tool")
    ts = _mod("transfer_scoring_tool")
    pp = _mod("performance_prediction_tool")

    def _clear():
        tr.set_retriever(None)
        es.set_embedder(None)
        es.set_task_repository(None)
        ts.set_transfer_strategies({})
        ts.set_task_repository(None)
        pp.set_orcamind_client(None)
        pp.set_task_repository(None)

    _clear()
    yield
    _clear()


class TestToolDocstrings:
    """Each tool must have a docstring — LangChain uses it as the tool description."""

    def test_task_retrieval_tool_has_docstring(self) -> None:
        from orcanet.reasoning.tools.task_retrieval_tool import task_retrieval_tool

        assert task_retrieval_tool.description

    def test_embedding_similarity_tool_has_docstring(self) -> None:
        from orcanet.reasoning.tools.embedding_similarity_tool import embedding_similarity_tool

        assert embedding_similarity_tool.description

    def test_transfer_scoring_tool_has_docstring(self) -> None:
        from orcanet.reasoning.tools.transfer_scoring_tool import transfer_scoring_tool

        assert transfer_scoring_tool.description

    def test_performance_prediction_tool_has_docstring(self) -> None:
        from orcanet.reasoning.tools.performance_prediction_tool import performance_prediction_tool

        assert performance_prediction_tool.description


class TestTaskRetrievalTool:
    @pytest.mark.asyncio
    async def test_returns_json_list_with_results(self, mock_retriever, sample_task) -> None:
        from orcanet.reasoning.tools.task_retrieval_tool import task_retrieval_tool

        _mod("task_retrieval_tool").set_retriever(mock_retriever)
        result = await task_retrieval_tool.ainvoke({"query": "brain MRI classification"})
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["task_id"] == str(sample_task.task_id)
        assert "score" in parsed[0]
        assert "reason" in parsed[0]

    @pytest.mark.asyncio
    async def test_returns_error_json_when_not_configured(self) -> None:
        from orcanet.reasoning.tools.task_retrieval_tool import task_retrieval_tool

        _mod("task_retrieval_tool").set_retriever(None)
        result = await task_retrieval_tool.ainvoke({"query": "some query"})
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_filters_applied_post_retrieval(self, mock_retriever, sample_task) -> None:
        from orcanet.reasoning.tools.task_retrieval_tool import task_retrieval_tool

        _mod("task_retrieval_tool").set_retriever(mock_retriever)
        filters = json.dumps({"domain": "nlp"})
        result = await task_retrieval_tool.ainvoke({"query": "query", "filters": filters})
        parsed = json.loads(result)
        assert parsed == []

    @pytest.mark.asyncio
    async def test_default_empty_filters_pass_all(self, mock_retriever, sample_task) -> None:
        from orcanet.reasoning.tools.task_retrieval_tool import task_retrieval_tool

        _mod("task_retrieval_tool").set_retriever(mock_retriever)
        result = await task_retrieval_tool.ainvoke({"query": "vision task"})
        parsed = json.loads(result)
        assert len(parsed) == 1


class TestEmbeddingSimilarityTool:
    @pytest.mark.asyncio
    async def test_returns_similarity_float(
        self, mock_embedder, mock_task_repository, sample_task, another_task
    ) -> None:
        from orcanet.reasoning.tools.embedding_similarity_tool import embedding_similarity_tool

        mock_embedder.embed = lambda x: torch.nn.functional.normalize(
            torch.ones(1, 64), dim=-1
        ).squeeze(0)
        em = _mod("embedding_similarity_tool")
        em.set_embedder(mock_embedder)
        em.set_task_repository(mock_task_repository)

        result = await embedding_similarity_tool.ainvoke(
            {"task_id_a": str(sample_task.task_id), "task_id_b": str(another_task.task_id)}
        )
        parsed = json.loads(result)
        assert "similarity" in parsed
        assert -1.0 <= parsed["similarity"] <= 1.0

    @pytest.mark.asyncio
    async def test_returns_error_when_not_configured(self) -> None:
        from orcanet.reasoning.tools.embedding_similarity_tool import embedding_similarity_tool
        from uuid import uuid4

        em = _mod("embedding_similarity_tool")
        em.set_embedder(None)
        em.set_task_repository(None)
        result = await embedding_similarity_tool.ainvoke(
            {"task_id_a": str(uuid4()), "task_id_b": str(uuid4())}
        )
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_returns_error_when_task_not_found(
        self, mock_embedder, mock_task_repository
    ) -> None:
        from uuid import uuid4
        from unittest.mock import AsyncMock
        from orcanet.reasoning.tools.embedding_similarity_tool import embedding_similarity_tool

        mock_task_repository.get_by_id = AsyncMock(return_value=None)
        em = _mod("embedding_similarity_tool")
        em.set_embedder(mock_embedder)
        em.set_task_repository(mock_task_repository)

        result = await embedding_similarity_tool.ainvoke(
            {"task_id_a": str(uuid4()), "task_id_b": str(uuid4())}
        )
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_similarity_is_cosine_regardless_of_embedder_scale(
        self, mock_embedder, mock_task_repository, sample_task, another_task
    ) -> None:
        from orcanet.reasoning.tools.embedding_similarity_tool import embedding_similarity_tool

        mock_embedder.embed = lambda x: torch.ones(1, 64) * 5.0
        em = _mod("embedding_similarity_tool")
        em.set_embedder(mock_embedder)
        em.set_task_repository(mock_task_repository)

        result = await embedding_similarity_tool.ainvoke(
            {"task_id_a": str(sample_task.task_id), "task_id_b": str(another_task.task_id)}
        )
        parsed = json.loads(result)
        assert parsed["similarity"] == pytest.approx(1.0)


class TestTransferScoringTool:
    @pytest.mark.asyncio
    async def test_returns_score_with_feature_strategy(
        self, mock_transfer_strategy, mock_task_repository, sample_task, another_task
    ) -> None:
        from orcanet.reasoning.tools.transfer_scoring_tool import transfer_scoring_tool

        ts = _mod("transfer_scoring_tool")
        ts.set_transfer_strategies({"feature": mock_transfer_strategy})
        ts.set_task_repository(mock_task_repository)

        result = await transfer_scoring_tool.ainvoke(
            {
                "source_task_id": str(sample_task.task_id),
                "target_task_id": str(another_task.task_id),
                "strategy": "feature",
            }
        )
        parsed = json.loads(result)
        assert "overall" in parsed
        assert parsed["overall"] == pytest.approx(0.75)
        assert "recommended_layers" in parsed

    @pytest.mark.asyncio
    async def test_unknown_strategy_returns_error(
        self, mock_transfer_strategy, mock_task_repository, sample_task, another_task
    ) -> None:
        from orcanet.reasoning.tools.transfer_scoring_tool import transfer_scoring_tool

        ts = _mod("transfer_scoring_tool")
        ts.set_transfer_strategies({"feature": mock_transfer_strategy})
        ts.set_task_repository(mock_task_repository)

        result = await transfer_scoring_tool.ainvoke(
            {
                "source_task_id": str(sample_task.task_id),
                "target_task_id": str(another_task.task_id),
                "strategy": "nonexistent",
            }
        )
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_returns_error_when_not_configured(self) -> None:
        from orcanet.reasoning.tools.transfer_scoring_tool import transfer_scoring_tool
        from uuid import uuid4

        ts = _mod("transfer_scoring_tool")
        ts.set_transfer_strategies({})
        ts.set_task_repository(None)
        result = await transfer_scoring_tool.ainvoke(
            {"source_task_id": str(uuid4()), "target_task_id": str(uuid4())}
        )
        parsed = json.loads(result)
        assert "error" in parsed


class TestPerformancePredictionTool:
    @pytest.mark.asyncio
    async def test_returns_predicted_metrics(
        self, mock_orcamind_client, mock_task_repository, sample_task
    ) -> None:
        from uuid import uuid4
        from orcanet.reasoning.tools.performance_prediction_tool import performance_prediction_tool

        pp = _mod("performance_prediction_tool")
        pp.set_orcamind_client(mock_orcamind_client)
        pp.set_task_repository(mock_task_repository)

        model_config = {"model_id": str(uuid4())}
        result = await performance_prediction_tool.ainvoke(
            {
                "task_id": str(sample_task.task_id),
                "model_config_json": json.dumps(model_config),
            }
        )
        parsed = json.loads(result)
        assert "metrics" in parsed
        assert "predicted_score" in parsed["metrics"]

    @pytest.mark.asyncio
    async def test_returns_error_when_not_configured(self) -> None:
        from orcanet.reasoning.tools.performance_prediction_tool import performance_prediction_tool
        from uuid import uuid4

        pp = _mod("performance_prediction_tool")
        pp.set_orcamind_client(None)
        pp.set_task_repository(None)
        result = await performance_prediction_tool.ainvoke(
            {"task_id": str(uuid4()), "model_config_json": "{}"}
        )
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_returns_error_for_missing_task(
        self, mock_orcamind_client, mock_task_repository
    ) -> None:
        from uuid import uuid4
        from unittest.mock import AsyncMock
        from orcanet.reasoning.tools.performance_prediction_tool import performance_prediction_tool

        mock_task_repository.get_by_id = AsyncMock(return_value=None)
        pp = _mod("performance_prediction_tool")
        pp.set_orcamind_client(mock_orcamind_client)
        pp.set_task_repository(mock_task_repository)

        result = await performance_prediction_tool.ainvoke(
            {"task_id": str(uuid4()), "model_config_json": "{}"}
        )
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_returns_error_for_non_object_json(
        self, mock_orcamind_client, mock_task_repository, sample_task
    ) -> None:
        from orcanet.reasoning.tools.performance_prediction_tool import performance_prediction_tool

        pp = _mod("performance_prediction_tool")
        pp.set_orcamind_client(mock_orcamind_client)
        pp.set_task_repository(mock_task_repository)

        for bad_json in ("[1,2,3]", '"just a string"', "42", "true"):
            result = await performance_prediction_tool.ainvoke(
                {"task_id": str(sample_task.task_id), "model_config_json": bad_json}
            )
            parsed = json.loads(result)
            assert "error" in parsed, f"Expected error for model_config_json={bad_json!r}"
