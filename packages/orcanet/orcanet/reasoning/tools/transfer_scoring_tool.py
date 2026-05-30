"""LangChain tool for scoring transferability between two tasks."""

from __future__ import annotations

import json
import logging
from uuid import UUID

logger = logging.getLogger(__name__)

_transfer_strategies: dict = {}
_task_repository = None


def set_transfer_strategies(strategies: dict) -> None:
    """Set the module-level strategy registry used by ``transfer_scoring_tool``."""
    global _transfer_strategies
    _transfer_strategies = strategies


def set_task_repository(repo) -> None:
    """Set the module-level task repository used by ``transfer_scoring_tool``."""
    global _task_repository
    _task_repository = repo


def get_transfer_strategies() -> dict:
    """Return the currently configured module-level transfer strategy registry."""
    return _transfer_strategies


def get_task_repository():
    """Return the currently configured module-level task repository (may be ``None``)."""
    return _task_repository


from langchain_core.tools import StructuredTool, tool


async def _run_transfer_scoring(
    source_task_id: str,
    target_task_id: str,
    strategy: str = "feature",
    *,
    transfer_strategies: dict,
    task_repository,
) -> str:
    if not transfer_strategies or task_repository is None:
        return json.dumps({"error": "Transfer strategies or task repository not configured"})
    try:
        scorer = transfer_strategies.get(strategy)
        if scorer is None:
            available = list(transfer_strategies.keys())
            return json.dumps({"error": f"Unknown strategy '{strategy}'. Available: {available}"})

        source_task = await task_repository.get_by_id(UUID(source_task_id))
        target_task = await task_repository.get_by_id(UUID(target_task_id))
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


@tool
async def transfer_scoring_tool(
    source_task_id: str,
    target_task_id: str,
    strategy: str = "feature",
) -> str:
    """Score the transferability between two tasks using the specified strategy."""
    return await _run_transfer_scoring(
        source_task_id,
        target_task_id,
        strategy,
        transfer_strategies=_transfer_strategies,
        task_repository=_task_repository,
    )


def make_transfer_scoring_tool(transfer_strategies: dict, task_repository) -> StructuredTool:
    """Return a new StructuredTool instance bound to the given strategies and repository."""

    async def _run(
        source_task_id: str, target_task_id: str, strategy: str = "feature"
    ) -> str:
        return await _run_transfer_scoring(
            source_task_id,
            target_task_id,
            strategy,
            transfer_strategies=transfer_strategies,
            task_repository=task_repository,
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="transfer_scoring_tool",
        description=transfer_scoring_tool.description,
    )
