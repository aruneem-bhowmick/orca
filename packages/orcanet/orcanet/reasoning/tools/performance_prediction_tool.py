"""LangChain tool for predicting model performance on a given task."""

from __future__ import annotations

import json
import logging
import math
from uuid import UUID

import numpy as np

logger = logging.getLogger(__name__)

_orcamind_client = None
_task_repository = None


def set_orcamind_client(client) -> None:
    """Set the module-level OrcaMind client used by ``performance_prediction_tool``."""
    global _orcamind_client
    _orcamind_client = client


def set_task_repository(repo) -> None:
    """Set the module-level task repository used by ``performance_prediction_tool``."""
    global _task_repository
    _task_repository = repo


def get_orcamind_client():
    """Return the currently configured module-level OrcaMind client (may be ``None``)."""
    return _orcamind_client


def get_task_repository():
    """Return the currently configured module-level task repository (may be ``None``)."""
    return _task_repository


def _task_to_embedding(task) -> list[float]:
    """Convert *task* statistics into a 25-dim float list for OrcaMind performance prediction."""
    vec = np.zeros(25, dtype=np.float32)
    if task.n_samples is not None:
        vec[0] = math.log1p(float(task.n_samples))
    if task.n_features is not None:
        vec[1] = float(task.n_features)
    if task.n_classes is not None:
        vec[2] = float(task.n_classes)
    return vec.tolist()


from langchain_core.tools import StructuredTool, tool


async def _run_performance_prediction(
    task_id: str, model_config_json: str, *, orcamind_client, task_repository
) -> str:
    """Parse model config JSON, resolve the task, and call OrcaMind for predicted metrics."""
    if orcamind_client is None or task_repository is None:
        return json.dumps({"error": "OrcaMind client or task repository not configured"})
    try:
        model_config = json.loads(model_config_json)
        if not isinstance(model_config, dict):
            return json.dumps({"error": "model_config_json must be a JSON object"})
        task = await task_repository.get_by_id(UUID(task_id))
        if task is None:
            return json.dumps({"error": f"Task {task_id} not found"})

        task_embedding = _task_to_embedding(task)
        model_id = UUID(model_config.get("model_id", str(UUID(int=0))))
        metrics = await orcamind_client.predict_performance(task_embedding, model_id)
        return json.dumps(
            {
                "task_id": task_id,
                "model_id": str(model_id),
                "metrics": metrics.final_metrics,
            }
        )
    except Exception as exc:
        logger.warning("performance_prediction_tool failed: %s", exc)
        return json.dumps({"error": str(exc)})


@tool
async def performance_prediction_tool(task_id: str, model_config_json: str) -> str:
    """Predict the performance of a model configuration on a given task."""
    return await _run_performance_prediction(
        task_id, model_config_json,
        orcamind_client=_orcamind_client,
        task_repository=_task_repository,
    )


def make_performance_prediction_tool(orcamind_client, task_repository) -> StructuredTool:
    """Return a new StructuredTool instance bound to the given client and repository."""

    async def _run(task_id: str, model_config_json: str) -> str:
        """Delegate to ``_run_performance_prediction`` with captured closure dependencies."""
        return await _run_performance_prediction(
            task_id, model_config_json,
            orcamind_client=orcamind_client,
            task_repository=task_repository,
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="performance_prediction_tool",
        description=performance_prediction_tool.description,
    )
