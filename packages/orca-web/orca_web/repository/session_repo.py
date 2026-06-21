"""Session / refresh-token management repository."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from orca_web.models.user import UserSession


class SessionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        user_id: uuid.UUID,
        jti: str,
        expires_at: datetime,
        device_info: str | None = None,
        ip_address: str | None = None,
    ) -> UserSession:
        row = UserSession(
            session_id=uuid.uuid4(),
            user_id=user_id,
            jti=jti,
            expires_at=expires_at,
            device_info=device_info,
            ip_address=ip_address,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def get_by_jti(self, jti: str) -> UserSession | None:
        result = await self._session.execute(
            select(UserSession).where(UserSession.jti == jti)
        )
        return result.scalar_one_or_none()

    async def revoke(self, jti: str) -> None:
        await self._session.execute(
            update(UserSession).where(UserSession.jti == jti).values(revoked=True)
        )
        await self._session.flush()

    async def revoke_all_for_user(self, user_id: uuid.UUID) -> None:
        await self._session.execute(
            update(UserSession)
            .where(UserSession.user_id == user_id, UserSession.revoked.is_(False))
            .values(revoked=True)
        )
        await self._session.flush()
