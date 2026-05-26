"""LLM-based re-ranker for ordering candidate tasks by relevance."""

from __future__ import annotations

import json
import logging

from langchain_core.language_models import BaseLLM
from pydantic import BaseModel, Field, ValidationError

from orca_shared.schemas.task import Task

logger = logging.getLogger(__name__)

_RERANK_PROMPT_TEMPLATE = """\
You are ranking candidate ML tasks by their relevance to a query task.

Query task:
  name: {query_name}
  domain: {query_domain}
  task_type: {query_task_type}
  n_samples: {query_n_samples}
  n_features: {query_n_features}
  n_classes: {query_n_classes}

Candidate tasks:
{candidates_text}

Return ONLY a JSON object with this structure (no markdown, no explanation):
{{
  "rankings": [
    {{"task_id": "<id>", "score": <0.0-1.0>, "reasoning": "<one sentence>"}},
    ...
  ]
}}

Order rankings from most to least relevant. Include all {n_candidates} candidates.
"""


class _RankedItem(BaseModel):
    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class _RankedList(BaseModel):
    rankings: list[_RankedItem]


class LLMRanker:
    """Re-ranks candidate tasks via LLM with Pydantic-validated JSON output."""

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    async def rerank(
        self,
        query_task: Task,
        candidate_tasks: list[Task],
        top_k: int = 10,
    ) -> list[tuple[Task, float, str]]:
        """Return up to *top_k* (task, score, reasoning) tuples sorted by score descending."""
        if not candidate_tasks:
            return []

        candidates_text = "\n".join(
            f"  [{t.task_id}] {t.name} | domain={t.domain} | type={t.task_type}"
            f" | samples={t.n_samples} | features={t.n_features} | classes={t.n_classes}"
            for t in candidate_tasks
        )
        prompt = _RERANK_PROMPT_TEMPLATE.format(
            query_name=query_task.name,
            query_domain=query_task.domain,
            query_task_type=query_task.task_type,
            query_n_samples=query_task.n_samples,
            query_n_features=query_task.n_features,
            query_n_classes=query_task.n_classes,
            candidates_text=candidates_text,
            n_candidates=len(candidate_tasks),
        )
        response = await self._llm.ainvoke(prompt)
        text = response if isinstance(response, str) else getattr(response, "content", "")
        ranked = _parse_ranked_list(text, candidate_tasks)
        return sorted(ranked, key=lambda x: x[1], reverse=True)[:top_k]


def _parse_ranked_list(
    text: str,
    candidate_tasks: list[Task],
) -> list[tuple[Task, float, str]]:
    """Parse LLM JSON response into (Task, score, reasoning) tuples."""
    task_by_id = {str(t.task_id): t for t in candidate_tasks}
    try:
        raw = text.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.splitlines()[1:])
            if raw.endswith("```"):
                raw = raw[:-3].strip()
        ranked = _RankedList.model_validate_json(raw)
    except (ValidationError, json.JSONDecodeError, ValueError):
        logger.warning("LLMRanker: failed to parse LLM response; returning empty ranking")
        return []

    results = []
    for item in ranked.rankings:
        task = task_by_id.get(item.task_id)
        if task is not None:
            results.append((task, item.score, item.reasoning))
    return results
