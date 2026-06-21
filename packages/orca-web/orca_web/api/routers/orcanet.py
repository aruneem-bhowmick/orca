"""Proxy router for OrcaNet service endpoints.

Forwards authenticated browser requests to the upstream OrcaNet service,
injecting an ``X-Orca-User-ID`` header and logging all operations to the
activity log.  Covers transfer scoring, transfer recommendations, task
retrieval, and transfer explanation.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from orca_web.api.deps import get_current_user, get_history_repo
from orca_web.api.proxy_utils import log_proxy_activity, proxy_request
from orca_web.config import settings
from orca_web.models.user import User
from orca_web.repository.history_repo import HistoryRepository

router = APIRouter(prefix="/orcanet", tags=["orcanet"])


@router.post("/transfer/score")
async def score_transfer(
    request: Request,
    user: User = Depends(get_current_user),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> Response:
    """Score transfer between two tasks via OrcaNet.

    Forwards the request body to OrcaNet's ``POST /api/v1/transfer/score``
    endpoint and logs a ``transfer_scored`` activity entry.
    """
    url = f"{settings.orcanet_api_url}/api/v1/transfer/score"
    response = await proxy_request(request=request, method="POST", target_url=url, user=user)
    await log_proxy_activity(
        history_repo=history_repo,
        user_id=user.user_id,
        action="transfer_scored",
        resource_type="transfer",
        service="orcanet",
        response=response,
    )
    return response


@router.post("/transfer/recommend")
async def recommend_transfer(
    request: Request,
    user: User = Depends(get_current_user),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> Response:
    """Get transfer recommendations from OrcaNet.

    Forwards to OrcaNet's ``POST /api/v1/transfer/recommend`` and logs a
    ``transfer_recommended`` activity entry.
    """
    url = f"{settings.orcanet_api_url}/api/v1/transfer/recommend"
    response = await proxy_request(request=request, method="POST", target_url=url, user=user)
    await log_proxy_activity(
        history_repo=history_repo,
        user_id=user.user_id,
        action="transfer_recommended",
        resource_type="transfer",
        service="orcanet",
        response=response,
    )
    return response


@router.post("/retrieve")
async def retrieve_tasks(
    request: Request,
    user: User = Depends(get_current_user),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> Response:
    """Retrieve similar tasks via OrcaNet.

    Forwards to OrcaNet's ``POST /api/v1/retrieve`` and logs a
    ``tasks_retrieved`` activity entry.
    """
    url = f"{settings.orcanet_api_url}/api/v1/retrieve"
    response = await proxy_request(request=request, method="POST", target_url=url, user=user)
    await log_proxy_activity(
        history_repo=history_repo,
        user_id=user.user_id,
        action="tasks_retrieved",
        resource_type="task",
        service="orcanet",
        response=response,
    )
    return response


@router.post("/explain")
async def explain_transfer(
    request: Request,
    user: User = Depends(get_current_user),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> Response:
    """Generate a transfer explanation via OrcaNet.

    Forwards to OrcaNet's ``POST /api/v1/explain`` and logs a
    ``transfer_explained`` activity entry.
    """
    url = f"{settings.orcanet_api_url}/api/v1/explain"
    response = await proxy_request(request=request, method="POST", target_url=url, user=user)
    await log_proxy_activity(
        history_repo=history_repo,
        user_id=user.user_id,
        action="transfer_explained",
        resource_type="transfer",
        service="orcanet",
        response=response,
    )
    return response
