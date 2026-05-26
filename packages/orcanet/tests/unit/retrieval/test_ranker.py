"""Unit tests for LLMRanker and _parse_ranked_list."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from orca_shared.schemas.task import Task
from orcanet.retrieval.ranker import LLMRanker, _parse_ranked_list

_NOW = datetime(2024, 1, 1)


def _make_task(**overrides) -> Task:
    defaults = dict(
        task_id=uuid4(),
        name="test-task",
        domain="vision",
        task_type="classification",
        n_samples=1000,
        n_features=10,
        n_classes=3,
        created_at=_NOW,
        updated_at=_NOW,
    )
    defaults.update(overrides)
    return Task(**defaults)


def _make_llm(content: str):
    response = MagicMock()
    response.content = content
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=response)
    return llm


# ---------------------------------------------------------------------------
# _parse_ranked_list
# ---------------------------------------------------------------------------


class TestParseRankedList:
    def test_valid_json_returns_correct_tuple(self) -> None:
        task = _make_task()
        tid = str(task.task_id)
        payload = json.dumps(
            {"rankings": [{"task_id": tid, "score": 0.9, "reasoning": "good match"}]}
        )
        result = _parse_ranked_list(payload, [task])
        assert len(result) == 1
        assert result[0][0] is task
        assert result[0][1] == pytest.approx(0.9)
        assert result[0][2] == "good match"

    def test_unknown_task_id_excluded(self) -> None:
        task = _make_task()
        payload = json.dumps(
            {
                "rankings": [
                    {
                        "task_id": "00000000-0000-0000-0000-000000000000",
                        "score": 0.8,
                        "reasoning": "irrelevant",
                    }
                ]
            }
        )
        result = _parse_ranked_list(payload, [task])
        assert result == []

    def test_markdown_backticks_stripped(self) -> None:
        task = _make_task()
        tid = str(task.task_id)
        inner = json.dumps(
            {"rankings": [{"task_id": tid, "score": 0.7, "reasoning": "ok"}]}
        )
        payload = f"```json\n{inner}\n```"
        result = _parse_ranked_list(payload, [task])
        assert len(result) == 1
        assert result[0][1] == pytest.approx(0.7)

    def test_invalid_json_returns_empty_list(self) -> None:
        task = _make_task()
        result = _parse_ranked_list("this is not json at all", [task])
        assert result == []

    def test_score_out_of_range_returns_empty_list(self) -> None:
        task = _make_task()
        tid = str(task.task_id)
        payload = json.dumps(
            {"rankings": [{"task_id": tid, "score": 1.5, "reasoning": "bad score"}]}
        )
        result = _parse_ranked_list(payload, [task])
        assert result == []

    def test_multiple_tasks_all_returned(self) -> None:
        tasks = [_make_task(name=f"task-{i}") for i in range(3)]
        rankings = [
            {"task_id": str(t.task_id), "score": 0.9 - 0.1 * i, "reasoning": f"r{i}"}
            for i, t in enumerate(tasks)
        ]
        payload = json.dumps({"rankings": rankings})
        result = _parse_ranked_list(payload, tasks)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# LLMRanker
# ---------------------------------------------------------------------------


class TestLLMRanker:
    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty_without_llm_call(self) -> None:
        llm = _make_llm("")
        ranker = LLMRanker(llm)
        result = await ranker.rerank(_make_task(), [])
        assert result == []
        llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_results_sorted_by_score_descending(self) -> None:
        tasks = [_make_task(name=f"t{i}") for i in range(3)]
        rankings = [
            {"task_id": str(tasks[0].task_id), "score": 0.5, "reasoning": "mid"},
            {"task_id": str(tasks[1].task_id), "score": 0.9, "reasoning": "high"},
            {"task_id": str(tasks[2].task_id), "score": 0.3, "reasoning": "low"},
        ]
        llm = _make_llm(json.dumps({"rankings": rankings}))
        ranker = LLMRanker(llm)
        result = await ranker.rerank(_make_task(), tasks)
        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_top_k_limits_results(self) -> None:
        tasks = [_make_task(name=f"t{i}") for i in range(5)]
        rankings = [
            {"task_id": str(t.task_id), "score": 0.9 - 0.1 * i, "reasoning": "r"}
            for i, t in enumerate(tasks)
        ]
        llm = _make_llm(json.dumps({"rankings": rankings}))
        ranker = LLMRanker(llm)
        result = await ranker.rerank(_make_task(), tasks, top_k=2)
        assert len(result) <= 2

    @pytest.mark.asyncio
    async def test_prompt_contains_query_task_metadata(self) -> None:
        query = _make_task(name="my-query-task", domain="nlp", task_type="ner")
        candidate = _make_task()
        payload = json.dumps(
            {
                "rankings": [
                    {"task_id": str(candidate.task_id), "score": 0.8, "reasoning": "ok"}
                ]
            }
        )
        llm = _make_llm(payload)
        ranker = LLMRanker(llm)
        await ranker.rerank(query, [candidate])
        prompt = llm.ainvoke.call_args[0][0]
        assert "my-query-task" in prompt
        assert "nlp" in prompt
        assert "ner" in prompt

    @pytest.mark.asyncio
    async def test_candidates_listed_in_prompt(self) -> None:
        candidates = [_make_task(name=f"cand-{i}") for i in range(2)]
        rankings = [
            {"task_id": str(t.task_id), "score": 0.8 - 0.1 * i, "reasoning": "ok"}
            for i, t in enumerate(candidates)
        ]
        llm = _make_llm(json.dumps({"rankings": rankings}))
        ranker = LLMRanker(llm)
        await ranker.rerank(_make_task(), candidates)
        prompt = llm.ainvoke.call_args[0][0]
        for c in candidates:
            assert str(c.task_id) in prompt
