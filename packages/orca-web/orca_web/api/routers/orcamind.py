"""Proxy router for OrcaMind service endpoints.

Forwards authenticated browser requests to the upstream OrcaMind service,
injecting an ``X-Orca-User-ID`` header and logging mutating operations to
the activity log.  GET endpoints pass through query parameters; POST
endpoints forward the request body and content-type.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from orca_web.api.deps import get_current_user, get_history_repo
from orca_web.api.proxy_utils import log_proxy_activity, proxy_request
from orca_web.config import settings
from orca_web.models.user import User
from orca_web.repository.history_repo import HistoryRepository

router = APIRouter(prefix="/orcamind", tags=["orcamind"])


@router.get("/tasks")
async def list_tasks(
    request: Request,
    user: User = Depends(get_current_user),
) -> Response:
    """List all registered tasks from OrcaMind."""
    url = f"{settings.orcamind_api_url}/api/v1/tasks"
    return await proxy_request(request=request, method="GET", target_url=url, user=user)


@router.get("/tasks/{task_id}")
async def get_task(
    task_id: str,
    request: Request,
    user: User = Depends(get_current_user),
) -> Response:
    """Retrieve a single task by ID from OrcaMind."""
    url = f"{settings.orcamind_api_url}/api/v1/tasks/{task_id}"
    return await proxy_request(request=request, method="GET", target_url=url, user=user)


@router.post("/tasks")
async def create_task(
    request: Request,
    user: User = Depends(get_current_user),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> Response:
    """Embed a new task via OrcaMind.

    Forwards the request body to OrcaMind's ``POST /api/v1/tasks/embed``
    endpoint and logs a ``task_created`` activity entry on completion.
    """
    url = f"{settings.orcamind_api_url}/api/v1/tasks/embed"
    response = await proxy_request(request=request, method="POST", target_url=url, user=user)
    await log_proxy_activity(
        history_repo=history_repo,
        user_id=user.user_id,
        action="task_created",
        resource_type="task",
        service="orcamind",
        response=response,
    )
    return response


@router.post("/recommend")
async def recommend_model(
    request: Request,
    user: User = Depends(get_current_user),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> Response:
    """Request model recommendations from OrcaMind.

    Forwards to OrcaMind's ``POST /api/v1/recommend-model`` and logs a
    ``model_recommended`` activity entry.
    """
    url = f"{settings.orcamind_api_url}/api/v1/recommend-model"
    response = await proxy_request(request=request, method="POST", target_url=url, user=user)
    await log_proxy_activity(
        history_repo=history_repo,
        user_id=user.user_id,
        action="model_recommended",
        resource_type="recommendation",
        service="orcamind",
        response=response,
    )
    return response


@router.post("/similar-tasks")
async def find_similar_tasks(
    request: Request,
    user: User = Depends(get_current_user),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> Response:
    """Search for similar tasks via OrcaMind.

    Forwards to OrcaMind's ``POST /api/v1/similar-tasks`` and logs a
    ``similar_tasks_searched`` activity entry.
    """
    url = f"{settings.orcamind_api_url}/api/v1/similar-tasks"
    response = await proxy_request(request=request, method="POST", target_url=url, user=user)
    await log_proxy_activity(
        history_repo=history_repo,
        user_id=user.user_id,
        action="similar_tasks_searched",
        resource_type="task",
        service="orcamind",
        response=response,
    )
    return response


@router.post("/predict-performance")
async def predict_performance(
    request: Request,
    user: User = Depends(get_current_user),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> Response:
    """Predict model performance via OrcaMind.

    Forwards to OrcaMind's ``POST /api/v1/predict-performance`` and logs
    a ``performance_predicted`` activity entry.
    """
    url = f"{settings.orcamind_api_url}/api/v1/predict-performance"
    response = await proxy_request(request=request, method="POST", target_url=url, user=user)
    await log_proxy_activity(
        history_repo=history_repo,
        user_id=user.user_id,
        action="performance_predicted",
        resource_type="prediction",
        service="orcamind",
        response=response,
    )
    return response
