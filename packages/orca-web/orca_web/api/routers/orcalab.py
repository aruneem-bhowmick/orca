"""Proxy router for OrcaLab service endpoints.

Forwards authenticated browser requests to the upstream OrcaLab service,
injecting an ``X-Orca-User-ID`` header and logging mutating operations to
the activity log.  Covers experiment CRUD and hyperparameter sweep
management.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from orca_web.api.deps import get_current_user, get_history_repo
from orca_web.api.proxy_utils import log_proxy_activity, proxy_request
from orca_web.config import settings
from orca_web.models.user import User
from orca_web.repository.history_repo import HistoryRepository

router = APIRouter(prefix="/orcalab", tags=["orcalab"])


@router.get("/experiments")
async def list_experiments(
    request: Request,
    user: User = Depends(get_current_user),
) -> Response:
    """List all experiments from OrcaLab."""
    url = f"{settings.orcalab_api_url}/api/v1/experiments"
    return await proxy_request(request=request, method="GET", target_url=url, user=user)


@router.get("/experiments/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    request: Request,
    user: User = Depends(get_current_user),
) -> Response:
    """Retrieve a single experiment by ID from OrcaLab."""
    url = f"{settings.orcalab_api_url}/api/v1/experiments/{experiment_id}"
    return await proxy_request(request=request, method="GET", target_url=url, user=user)


@router.post("/experiments")
async def create_experiment(
    request: Request,
    user: User = Depends(get_current_user),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> Response:
    """Create a new experiment via OrcaLab.

    Forwards the request body to OrcaLab's ``POST /api/v1/experiments``
    endpoint and logs an ``experiment_started`` activity entry on
    completion.
    """
    url = f"{settings.orcalab_api_url}/api/v1/experiments"
    response = await proxy_request(request=request, method="POST", target_url=url, user=user)
    await log_proxy_activity(
        history_repo=history_repo,
        user_id=user.user_id,
        action="experiment_started",
        resource_type="experiment",
        service="orcalab",
        response=response,
    )
    return response


@router.post("/sweeps")
async def create_sweep(
    request: Request,
    user: User = Depends(get_current_user),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> Response:
    """Start a hyperparameter sweep via OrcaLab.

    Forwards the request body to OrcaLab's ``POST /api/v1/sweeps``
    endpoint and logs a ``sweep_started`` activity entry on completion.
    """
    url = f"{settings.orcalab_api_url}/api/v1/sweeps"
    response = await proxy_request(request=request, method="POST", target_url=url, user=user)
    await log_proxy_activity(
        history_repo=history_repo,
        user_id=user.user_id,
        action="sweep_started",
        resource_type="sweep",
        service="orcalab",
        response=response,
    )
    return response


@router.get("/sweeps/{sweep_id}")
async def get_sweep(
    sweep_id: str,
    request: Request,
    user: User = Depends(get_current_user),
) -> Response:
    """Retrieve sweep status by ID from OrcaLab."""
    url = f"{settings.orcalab_api_url}/api/v1/sweeps/{sweep_id}"
    return await proxy_request(request=request, method="GET", target_url=url, user=user)
