"""Activity log and bookmark repository."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from orca_web.models.user import ActivityLog, UserBookmark


class HistoryRepository:
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

    async def list_global_feed(self, *, limit: int = 50, offset: int = 0) -> list[ActivityLog]:
        stmt = (
            select(ActivityLog)
            .order_by(desc(ActivityLog.created_at))
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    # ── Bookmarks ─────────────────────────────────────────────────────────

    async def add_bookmark(
        self,
        *,
        user_id: uuid.UUID,
        resource_type: str,
        resource_id: str,
        note: str | None = None,
    ) -> UserBookmark:
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

    async def delete_bookmark(self, bookmark_id: uuid.UUID, user_id: uuid.UUID) -> bool:
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
        stmt = (
            select(UserBookmark)
            .where(UserBookmark.user_id == user_id)
            .order_by(desc(UserBookmark.created_at))
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
