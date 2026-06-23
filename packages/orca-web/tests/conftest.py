"""Shared fixtures for orca-web tests."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

_REPO_ROOT = Path(__file__).parents[3]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the absolute path to the repository root directory."""
    return _REPO_ROOT


@pytest.fixture()
def mock_settings(monkeypatch):
    """Return a Settings-like namespace and patch it into all orca_web modules."""
    s = SimpleNamespace(
        database_url="postgresql+asyncpg://test:test@localhost/test",
        redis_url="redis://localhost:6379",
        jwt_secret_key="test-secret-key-for-unit-tests",
        jwt_algorithm="HS256",
        access_token_expire_minutes=15,
        refresh_token_expire_days=7,
        google_client_id="",
        google_client_secret="",
        github_client_id="",
        github_client_secret="",
        orcamind_api_url="http://localhost:8000",
        orcalab_api_url="http://localhost:8001",
        orcanet_api_url="http://localhost:8002",
        frontend_url="http://localhost:5173",
        cors_origins="http://localhost:5173,http://localhost:3000",
    )
    monkeypatch.setattr("orca_web.auth.jwt.settings", s)
    monkeypatch.setattr("orca_web.services.aggregator.settings", s)
    monkeypatch.setattr("orca_web.api.middleware.settings", s)
    monkeypatch.setattr("orca_web.api.routers.auth.settings", s)
    monkeypatch.setattr("orca_web.api.routers.orcamind.settings", s)
    monkeypatch.setattr("orca_web.api.routers.orcalab.settings", s)
    monkeypatch.setattr("orca_web.api.routers.orcanet.settings", s)
    monkeypatch.setattr("orca_web.api.main.settings", s)
    monkeypatch.setattr("orca_web.api.websocket.settings", s)
    return s


@pytest.fixture()
def user_factory():
    """Return a callable that builds User-like mock objects."""

    def _make(
        *,
        user_id=None,
        email="alice@example.com",
        username="alice",
        password_hash="$2b$12$hashedvalue",
        oauth_provider=None,
        oauth_sub=None,
        role="user",
        preferences=None,
        is_active=True,
    ):
        u = MagicMock()
        u.user_id = user_id or uuid.uuid4()
        u.email = email
        u.username = username
        u.password_hash = password_hash
        u.oauth_provider = oauth_provider
        u.oauth_sub = oauth_sub
        u.role = role
        u.preferences = preferences
        u.is_active = is_active
        u.created_at = datetime.now(timezone.utc)
        u.updated_at = datetime.now(timezone.utc)
        return u

    return _make


@pytest.fixture()
def mock_async_session():
    """Return an AsyncMock that behaves like an AsyncSession."""
    session = AsyncMock()
    session.flush = AsyncMock()
    session.add = MagicMock()

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None
    result_mock.scalars.return_value.all.return_value = []
    result_mock.rowcount = 0
    session.execute = AsyncMock(return_value=result_mock)

    return session
