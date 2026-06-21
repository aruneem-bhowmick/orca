"""User CRUD repository."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from orca_web.models.user import User


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        email: str,
        username: str,
        password_hash: str | None = None,
        oauth_provider: str | None = None,
        oauth_sub: str | None = None,
    ) -> User:
        user = User(
            user_id=uuid.uuid4(),
            email=email,
            username=username,
            password_hash=password_hash,
            oauth_provider=oauth_provider,
            oauth_sub=oauth_sub,
        )
        self._session.add(user)
        await self._session.flush()
        return user

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        result = await self._session.execute(select(User).where(User.user_id == user_id))
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        result = await self._session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_by_username(self, username: str) -> User | None:
        result = await self._session.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()

    async def get_by_oauth(self, provider: str, sub: str) -> User | None:
        result = await self._session.execute(
            select(User).where(User.oauth_provider == provider, User.oauth_sub == sub)
        )
        return result.scalar_one_or_none()

    async def update_profile(self, user_id: uuid.UUID, **fields: Any) -> None:
        await self._session.execute(
            update(User).where(User.user_id == user_id).values(**fields)
        )
        await self._session.flush()
