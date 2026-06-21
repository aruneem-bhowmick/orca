"""Tests for orca_web.api.deps."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from jose import jwt as jose_jwt

from orca_web.api.deps import get_current_user, get_optional_user

SECRET = "test-deps-secret"
ALGO = "HS256"


@pytest.fixture(autouse=True)
def _patch_jwt_settings(monkeypatch):
    s = SimpleNamespace(
        jwt_secret_key=SECRET,
        jwt_algorithm=ALGO,
        access_token_expire_minutes=15,
        refresh_token_expire_days=7,
    )
    monkeypatch.setattr("orca_web.auth.jwt.settings", s)


def _make_token(user_id, token_type="access", expired=False):
    exp = datetime.now(timezone.utc) + (
        timedelta(hours=-1) if expired else timedelta(hours=1)
    )
    return jose_jwt.encode(
        {"sub": str(user_id), "type": token_type, "exp": exp},
        SECRET,
        algorithm=ALGO,
    )


def _make_request(token=None):
    req = MagicMock()
    if token:
        req.headers = {"Authorization": f"Bearer {token}"}
    else:
        req.headers = {}
    return req


class TestGetCurrentUser:
    async def test_returns_user_for_valid_token(self, user_factory):
        uid = uuid.uuid4()
        user = user_factory(user_id=uid, is_active=True)
        token = _make_token(uid)
        request = _make_request(token)
        session = AsyncMock()

        with patch("orca_web.api.deps.UserRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=user)
            result = await get_current_user(request, session)
            assert result.user_id == uid

    async def test_raises_on_missing_header(self):
        request = _make_request(None)
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(request, AsyncMock())
        assert exc_info.value.status_code == 401

    async def test_raises_on_expired_token(self):
        token = _make_token(uuid.uuid4(), expired=True)
        request = _make_request(token)
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(request, AsyncMock())
        assert exc_info.value.status_code == 401

    async def test_raises_on_refresh_token_type(self, user_factory):
        token = _make_token(uuid.uuid4(), token_type="refresh")
        request = _make_request(token)
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(request, AsyncMock())
        assert exc_info.value.status_code == 401

    async def test_raises_on_inactive_user(self, user_factory):
        uid = uuid.uuid4()
        user = user_factory(user_id=uid, is_active=False)
        token = _make_token(uid)
        request = _make_request(token)
        session = AsyncMock()

        with patch("orca_web.api.deps.UserRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=user)
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(request, session)
            assert exc_info.value.status_code == 401

    async def test_raises_on_nonexistent_user(self):
        uid = uuid.uuid4()
        token = _make_token(uid)
        request = _make_request(token)
        session = AsyncMock()

        with patch("orca_web.api.deps.UserRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=None)
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(request, session)
            assert exc_info.value.status_code == 401


class TestGetOptionalUser:
    async def test_returns_user_for_valid_token(self, user_factory):
        uid = uuid.uuid4()
        user = user_factory(user_id=uid, is_active=True)
        token = _make_token(uid)
        request = _make_request(token)
        session = AsyncMock()

        with patch("orca_web.api.deps.UserRepository") as MockRepo:
            MockRepo.return_value.get_by_id = AsyncMock(return_value=user)
            result = await get_optional_user(request, session)
            assert result is not None
            assert result.user_id == uid

    async def test_returns_none_on_missing_header(self):
        request = _make_request(None)
        result = await get_optional_user(request, AsyncMock())
        assert result is None

    async def test_returns_none_on_expired_token(self):
        token = _make_token(uuid.uuid4(), expired=True)
        request = _make_request(token)
        result = await get_optional_user(request, AsyncMock())
        assert result is None

    async def test_returns_none_on_refresh_token_type(self):
        token = _make_token(uuid.uuid4(), token_type="refresh")
        request = _make_request(token)
        result = await get_optional_user(request, AsyncMock())
        assert result is None
