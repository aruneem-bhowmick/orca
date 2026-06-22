"""Activity log and bookmark repository."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import delete, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from orca_web.models.user import ActivityLog, UserBookmark


class HistoryRepository:
    """Async operations for activity logs and user bookmarks."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── Activity log ──────────────────────────────────────────────────────

    async def log_activity(
        self,
        *,
        user_id: uuid.UUID,
        action: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
        service: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> ActivityLog:
        """Insert an activity log entry and flush."""
        row = ActivityLog(
            log_id=uuid.uuid4(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            service=service,
            details=details,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def list_for_user(
        self,
        user_id: uuid.UUID,
        *,
        limit: int = 50,
        offset: int = 0,
        service: str | None = None,
        resource_type: str | None = None,
    ) -> list[ActivityLog]:
        """Return paginated activity entries for *user_id*, newest first."""
        stmt = (
            select(ActivityLog)
            .where(ActivityLog.user_id == user_id)
            .order_by(desc(ActivityLog.created_at))
        )
        if service:
            stmt = stmt.where(ActivityLog.service == service)
        if resource_type:
            stmt = stmt.where(ActivityLog.resource_type == resource_type)
        stmt = stmt.limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def count_for_user(
        self,
        user_id: uuid.UUID,
        *,
        service: str | None = None,
        resource_type: str | None = None,
    ) -> int:
        """Return the total number of activity entries matching the filters for *user_id*."""
        stmt = select(func.count(ActivityLog.log_id)).where(
            ActivityLog.user_id == user_id
        )
        if service:
            stmt = stmt.where(ActivityLog.service == service)
        if resource_type:
            stmt = stmt.where(ActivityLog.resource_type == resource_type)
        result = await self._session.execute(stmt)
        return result.scalar_one()

    async def list_global_feed(self, *, limit: int = 50, offset: int = 0) -> list[ActivityLog]:
        """Return a paginated cross-user activity feed, newest first."""
        stmt = (
            select(ActivityLog)
            .order_by(desc(ActivityLog.created_at))
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def count_global_feed(self) -> int:
        """Return the total number of entries in the global activity feed."""
        result = await self._session.execute(
            select(func.count(ActivityLog.log_id))
        )
        return result.scalar_one()

    # ── Bookmarks ─────────────────────────────────────────────────────────

    async def add_bookmark(
        self,
        *,
        user_id: uuid.UUID,
        resource_type: str,
        resource_id: str,
        note: str | None = None,
    ) -> UserBookmark:
        """Insert a bookmark and flush."""
        row = UserBookmark(
            bookmark_id=uuid.uuid4(),
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            note=note,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def get_bookmark_by_id(self, bookmark_id: uuid.UUID) -> UserBookmark | None:
        """Return the bookmark with the given ID regardless of owner, or ``None``."""
        result = await self._session.execute(
            select(UserBookmark).where(UserBookmark.bookmark_id == bookmark_id)
        )
        return result.scalar_one_or_none()

    async def delete_bookmark(self, bookmark_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """Delete a bookmark owned by *user_id*.  Returns ``True`` if a row was removed."""
        result = await self._session.execute(
            delete(UserBookmark).where(
                UserBookmark.bookmark_id == bookmark_id,
                UserBookmark.user_id == user_id,
            )
        )
        await self._session.flush()
        return result.rowcount > 0

    async def list_bookmarks(
        self, user_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> list[UserBookmark]:
        """Return paginated bookmarks for *user_id*, newest first."""
        stmt = (
            select(UserBookmark)
            .where(UserBookmark.user_id == user_id)
            .order_by(desc(UserBookmark.created_at))
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def count_bookmarks(self, user_id: uuid.UUID) -> int:
        """Return the total number of bookmarks owned by *user_id*."""
        result = await self._session.execute(
            select(func.count(UserBookmark.bookmark_id)).where(
                UserBookmark.user_id == user_id
            )
        )
        return result.scalar_one()
