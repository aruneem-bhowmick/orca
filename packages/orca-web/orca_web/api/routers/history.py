"""History and bookmark endpoints for user activity tracking."""

from __future__ import annotations

import math
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from orca_web.api.deps import get_current_user, get_history_repo
from orca_web.models.user import User
from orca_web.repository.history_repo import HistoryRepository

router = APIRouter(tags=["history"])


# ── Request / response schemas ────────────────────────────────────────────


class ActivityResponse(BaseModel):
    """Single activity log entry returned in paginated lists."""

    log_id: UUID
    user_id: UUID
    action: str
    resource_type: str | None = None
    resource_id: str | None = None
    service: str | None = None
    details: dict[str, Any] | None = None
    created_at: str | None = None

    model_config = {"from_attributes": True}


class PaginatedResponse(BaseModel):
    """Envelope for paginated list endpoints."""

    items: list[dict[str, Any]]
    total: int
    page: int
    per_page: int
    pages: int


class BookmarkResponse(BaseModel):
    """Single bookmark entry returned in lists and on creation."""

    bookmark_id: UUID
    user_id: UUID
    resource_type: str
    resource_id: str
    note: str | None = None
    created_at: str | None = None

    model_config = {"from_attributes": True}


class BookmarkCreateRequest(BaseModel):
    """Payload for creating a new bookmark."""

    resource_type: str = Field(..., min_length=1, max_length=50)
    resource_id: str = Field(..., min_length=1, max_length=255)
    note: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────


def _serialize_activity(row: Any) -> dict[str, Any]:
    """Convert an ActivityLog ORM instance to a JSON-compatible dict."""
    return {
        "log_id": str(row.log_id),
        "user_id": str(row.user_id),
        "action": row.action,
        "resource_type": row.resource_type,
        "resource_id": row.resource_id,
        "service": row.service,
        "details": row.details,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _serialize_bookmark(row: Any) -> dict[str, Any]:
    """Convert a UserBookmark ORM instance to a JSON-compatible dict."""
    return {
        "bookmark_id": str(row.bookmark_id),
        "user_id": str(row.user_id),
        "resource_type": row.resource_type,
        "resource_id": row.resource_id,
        "note": row.note,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _paginate(
    items: list[dict[str, Any]], *, total: int, page: int, per_page: int
) -> PaginatedResponse:
    """Build a paginated response envelope from pre-fetched items."""
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        per_page=per_page,
        pages=max(1, math.ceil(total / per_page)),
    )


# ── Activity log endpoints ────────────────────────────────────────────────


@router.get("/history", response_model=PaginatedResponse)
async def list_history(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> PaginatedResponse:
    """Return the paginated activity log for the current user."""
    offset = (page - 1) * per_page
    total = await history_repo.count_for_user(user.user_id)
    rows = await history_repo.list_for_user(
        user.user_id, limit=per_page, offset=offset
    )
    items = [_serialize_activity(r) for r in rows]
    return _paginate(items, total=total, page=page, per_page=per_page)


@router.get("/history/tasks", response_model=PaginatedResponse)
async def list_task_history(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> PaginatedResponse:
    """Return the paginated activity log filtered to OrcaMind (task) actions."""
    offset = (page - 1) * per_page
    total = await history_repo.count_for_user(user.user_id, service="orcamind")
    rows = await history_repo.list_for_user(
        user.user_id, limit=per_page, offset=offset, service="orcamind"
    )
    items = [_serialize_activity(r) for r in rows]
    return _paginate(items, total=total, page=page, per_page=per_page)


@router.get("/history/experiments", response_model=PaginatedResponse)
async def list_experiment_history(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> PaginatedResponse:
    """Return the paginated activity log filtered to OrcaLab (experiment) actions."""
    offset = (page - 1) * per_page
    total = await history_repo.count_for_user(user.user_id, service="orcalab")
    rows = await history_repo.list_for_user(
        user.user_id, limit=per_page, offset=offset, service="orcalab"
    )
    items = [_serialize_activity(r) for r in rows]
    return _paginate(items, total=total, page=page, per_page=per_page)


# ── Bookmark endpoints ────────────────────────────────────────────────────


@router.post("/bookmarks", status_code=status.HTTP_201_CREATED)
async def create_bookmark(
    body: BookmarkCreateRequest,
    user: User = Depends(get_current_user),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> dict[str, Any]:
    """Create a new bookmark for the current user."""
    bookmark = await history_repo.add_bookmark(
        user_id=user.user_id,
        resource_type=body.resource_type,
        resource_id=body.resource_id,
        note=body.note,
    )
    return _serialize_bookmark(bookmark)


@router.delete(
    "/bookmarks/{bookmark_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_bookmark(
    bookmark_id: UUID,
    user: User = Depends(get_current_user),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> None:
    """Delete a bookmark by ID.

    Returns 404 if the bookmark does not exist and 403 if the bookmark
    belongs to a different user.
    """
    existing = await history_repo.get_bookmark_by_id(bookmark_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bookmark not found",
        )
    if existing.user_id != user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot delete another user's bookmark",
        )
    await history_repo.delete_bookmark(bookmark_id, user.user_id)


@router.get("/bookmarks", response_model=PaginatedResponse)
async def list_bookmarks(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> PaginatedResponse:
    """Return the paginated bookmarks for the current user."""
    offset = (page - 1) * per_page
    total = await history_repo.count_bookmarks(user.user_id)
    rows = await history_repo.list_bookmarks(
        user.user_id, limit=per_page, offset=offset
    )
    items = [_serialize_bookmark(r) for r in rows]
    return _paginate(items, total=total, page=page, per_page=per_page)


# ── Global feed ───────────────────────────────────────────────────────────


@router.get("/feed", response_model=PaginatedResponse)
async def global_feed(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> PaginatedResponse:
    """Return the paginated global activity feed across all users."""
    offset = (page - 1) * per_page
    total = await history_repo.count_global_feed()
    rows = await history_repo.list_global_feed(limit=per_page, offset=offset)
    items = [_serialize_activity(r) for r in rows]
    return _paginate(items, total=total, page=page, per_page=per_page)
