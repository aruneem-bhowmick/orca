"""LangChain tool for scoring transferability between two tasks."""

from __future__ import annotations

import json
import logging
from uuid import UUID

logger = logging.getLogger(__name__)

_transfer_strategies: dict = {}
_task_repository = None


def set_transfer_strategies(strategies: dict) -> None:
    global _transfer_strategies
    _transfer_strategies = strategies


def set_task_repository(repo) -> None:
    global _task_repository
    _task_repository = repo


def get_transfer_strategies() -> dict:
    return _transfer_strategies


def get_task_repository():
    return _task_repository


from langchain_core.tools import tool


@tool
async def transfer_scoring_tool(
    source_task_id: str,
    target_task_id: str,
    strategy: str = "feature",
) -> str:
    """Score the transferability between two tasks using the specified strategy."""
    if not _transfer_strategies or _task_repository is None:
        return json.dumps({"error": "Transfer strategies or task repository not configured"})
    try:
        scorer = _transfer_strategies.get(strategy)
        if scorer is None:
            available = list(_transfer_strategies.keys())
            return json.dumps({"error": f"Unknown strategy '{strategy}'. Available: {available}"})

        source_task = await _task_repository.get_by_id(UUID(source_task_id))
        target_task = await _task_repository.get_by_id(UUID(target_task_id))
        if source_task is None or target_task is None:
            return json.dumps({"error": "One or both tasks not found"})

        score = scorer.score_transfer(source_task, target_task)
        return json.dumps(
            {
                "overall": score.overall,
                "layer_scores": score.layer_scores,
                "recommended_layers": score.recommended_layers,
                "reasoning": score.reasoning,
            }
        )
    except Exception as exc:
        logger.warning("transfer_scoring_tool failed: %s", exc)
        return json.dumps({"error": str(exc)})
