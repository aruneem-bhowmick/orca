"""FastAPI dependency providers."""

from __future__ import annotations

import uuid
from typing import AsyncGenerator

from fastapi import Cookie, Depends, HTTPException, Request, status
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from orca_web.auth.jwt import decode_token
from orca_web.models.user import User
from orca_web.repository.history_repo import HistoryRepository
from orca_web.repository.session_repo import SessionRepository
from orca_web.repository.user_repo import UserRepository
from orca_web.services.aggregator import Aggregator


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    async with request.app.state.db_sessionmaker() as session:
        async with session.begin():
            yield session


async def get_user_repo(session: AsyncSession = Depends(get_db)) -> UserRepository:
    return UserRepository(session)


async def get_session_repo(session: AsyncSession = Depends(get_db)) -> SessionRepository:
    return SessionRepository(session)


async def get_history_repo(session: AsyncSession = Depends(get_db)) -> HistoryRepository:
    return HistoryRepository(session)


def get_aggregator(request: Request) -> Aggregator:
    return Aggregator(request.app.state.http_client)


async def get_current_user(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> User:
    """Extract and validate the JWT from the Authorization header."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header",
        )
    token = auth_header.split(" ", 1)[1]
    try:
        payload = decode_token(token)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject",
        )

    repo = UserRepository(session)
    user = await repo.get_by_id(uuid.UUID(user_id))
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    return user


async def get_optional_user(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> User | None:
    """Like get_current_user but returns None instead of 401."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split(" ", 1)[1]
    try:
        payload = decode_token(token)
    except JWTError:
        return None
    if payload.get("type") != "access":
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    repo = UserRepository(session)
    return await repo.get_by_id(uuid.UUID(user_id))
