"""Tests for orca_web.api.routers.auth endpoints."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Response

from orca_web.api.routers.auth import (
    LoginRequest,
    ProfileUpdate,
    RegisterRequest,
    TokenResponse,
    _clear_refresh_cookie,
    _set_refresh_cookie,
    login,
    logout,
    get_me,
    oauth_callback,
    oauth_redirect,
    refresh_token,
    register,
    update_me,
)

SECRET = "test-auth-secret"
ALGO = "HS256"


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch):
    s = SimpleNamespace(
        jwt_secret_key=SECRET,
        jwt_algorithm=ALGO,
        access_token_expire_minutes=15,
        refresh_token_expire_days=7,
        frontend_url="http://localhost:5173",
    )
    monkeypatch.setattr("orca_web.auth.jwt.settings", s)
    monkeypatch.setattr("orca_web.api.routers.auth.settings", s)


class TestSetRefreshCookie:
    def test_sets_httponly_cookie(self):
        resp = Response()
        _set_refresh_cookie(resp, "tok-123")
        raw = resp.headers.get("set-cookie", "")
        assert "refresh_token=" in raw
        assert "httponly" in raw.lower()

    def test_clear_deletes_cookie(self):
        resp = Response()
        _clear_refresh_cookie(resp)
        raw = resp.headers.get("set-cookie", "")
        assert "refresh_token=" in raw


class TestRegister:
    async def test_success(self, user_factory):
        user = user_factory(email="new@x.com", username="newuser")
        user_repo = AsyncMock()
        user_repo.get_by_email = AsyncMock(return_value=None)
        user_repo.get_by_username = AsyncMock(return_value=None)
        user_repo.create = AsyncMock(return_value=user)
        session_repo = AsyncMock()
        session_repo.create = AsyncMock()
        resp = Response()
        body = RegisterRequest(email="new@x.com", username="newuser", password="pw")
        result = await register(body, resp, user_repo, session_repo)
        assert isinstance(result, TokenResponse)
        assert result.access_token
        user_repo.create.assert_awaited_once()

    async def test_duplicate_email_returns_409(self, user_factory):
        user_repo = AsyncMock()
        user_repo.get_by_email = AsyncMock(return_value=user_factory())
        body = RegisterRequest(email="dup@x.com", username="u", password="p")
        with pytest.raises(HTTPException) as exc:
            await register(body, Response(), user_repo, AsyncMock())
        assert exc.value.status_code == 409

    async def test_duplicate_username_returns_409(self, user_factory):
        user_repo = AsyncMock()
        user_repo.get_by_email = AsyncMock(return_value=None)
        user_repo.get_by_username = AsyncMock(return_value=user_factory())
        body = RegisterRequest(email="ok@x.com", username="taken", password="p")
        with pytest.raises(HTTPException) as exc:
            await register(body, Response(), user_repo, AsyncMock())
        assert exc.value.status_code == 409


class TestLogin:
    async def test_success(self, user_factory):
        from orca_web.auth.password import hash_password

        hashed = hash_password("correct")
        user = user_factory(password_hash=hashed)
        user_repo = AsyncMock()
        user_repo.get_by_email = AsyncMock(return_value=user)
        session_repo = AsyncMock()
        session_repo.create = AsyncMock()
        body = LoginRequest(email="alice@example.com", password="correct")
        result = await login(body, Response(), user_repo, session_repo)
        assert result.access_token

    async def test_wrong_password_returns_401(self, user_factory):
        from orca_web.auth.password import hash_password

        user = user_factory(password_hash=hash_password("right"))
        user_repo = AsyncMock()
        user_repo.get_by_email = AsyncMock(return_value=user)
        body = LoginRequest(email="alice@example.com", password="wrong")
        with pytest.raises(HTTPException) as exc:
            await login(body, Response(), user_repo, AsyncMock())
        assert exc.value.status_code == 401

    async def test_unknown_email_returns_401(self):
        user_repo = AsyncMock()
        user_repo.get_by_email = AsyncMock(return_value=None)
        body = LoginRequest(email="no@x.com", password="pw")
        with pytest.raises(HTTPException) as exc:
            await login(body, Response(), user_repo, AsyncMock())
        assert exc.value.status_code == 401

    async def test_oauth_user_without_password_returns_401(self, user_factory):
        user = user_factory(password_hash=None, oauth_provider="google")
        user_repo = AsyncMock()
        user_repo.get_by_email = AsyncMock(return_value=user)
        body = LoginRequest(email="alice@example.com", password="pw")
        with pytest.raises(HTTPException) as exc:
            await login(body, Response(), user_repo, AsyncMock())
        assert exc.value.status_code == 401


class TestRefresh:
    async def test_success(self, user_factory):
        from orca_web.auth.jwt import create_refresh_token

        uid = uuid.uuid4()
        token, jti, expires = create_refresh_token(str(uid))
        user = user_factory(user_id=uid)
        session_mock = MagicMock()
        session_mock.revoked = False
        session_repo = AsyncMock()
        session_repo.get_by_jti = AsyncMock(return_value=session_mock)
        session_repo.revoke = AsyncMock()
        session_repo.create = AsyncMock()
        user_repo = AsyncMock()
        user_repo.get_by_id = AsyncMock(return_value=user)
        request = MagicMock()
        request.cookies = {"refresh_token": token}
        result = await refresh_token(request, Response(), user_repo, session_repo)
        assert result.access_token
        session_repo.revoke.assert_awaited_once_with(jti)

    async def test_missing_cookie_returns_401(self):
        request = MagicMock()
        request.cookies = {}
        with pytest.raises(HTTPException) as exc:
            await refresh_token(request, Response(), AsyncMock(), AsyncMock())
        assert exc.value.status_code == 401

    async def test_revoked_session_returns_401(self):
        from orca_web.auth.jwt import create_refresh_token

        token, jti, _ = create_refresh_token(str(uuid.uuid4()))
        session_mock = MagicMock()
        session_mock.revoked = True
        session_repo = AsyncMock()
        session_repo.get_by_jti = AsyncMock(return_value=session_mock)
        request = MagicMock()
        request.cookies = {"refresh_token": token}
        with pytest.raises(HTTPException) as exc:
            await refresh_token(request, Response(), AsyncMock(), session_repo)
        assert exc.value.status_code == 401

    async def test_invalid_token_returns_401(self):
        request = MagicMock()
        request.cookies = {"refresh_token": "garbage.token.here"}
        with pytest.raises(HTTPException) as exc:
            await refresh_token(request, Response(), AsyncMock(), AsyncMock())
        assert exc.value.status_code == 401

    async def test_inactive_user_returns_401(self, user_factory):
        from orca_web.auth.jwt import create_refresh_token

        uid = uuid.uuid4()
        token, jti, _ = create_refresh_token(str(uid))
        user = user_factory(user_id=uid, is_active=False)
        session_mock = MagicMock()
        session_mock.revoked = False
        session_repo = AsyncMock()
        session_repo.get_by_jti = AsyncMock(return_value=session_mock)
        user_repo = AsyncMock()
        user_repo.get_by_id = AsyncMock(return_value=user)
        request = MagicMock()
        request.cookies = {"refresh_token": token}
        with pytest.raises(HTTPException) as exc:
            await refresh_token(request, Response(), user_repo, session_repo)
        assert exc.value.status_code == 401


class TestLogout:
    async def test_revokes_and_clears(self):
        from orca_web.auth.jwt import create_refresh_token

        token, jti, _ = create_refresh_token("user-1")
        session_repo = AsyncMock()
        session_repo.revoke = AsyncMock()
        request = MagicMock()
        request.cookies = {"refresh_token": token}
        resp = Response()
        await logout(request, resp, session_repo)
        session_repo.revoke.assert_awaited_once_with(jti)

    async def test_no_cookie_still_clears(self):
        request = MagicMock()
        request.cookies = {}
        await logout(request, Response(), AsyncMock())


class TestOAuthRedirect:
    async def test_unknown_provider_returns_400(self):
        request = MagicMock()
        with patch("orca_web.api.routers.auth.oauth") as mock_oauth:
            mock_oauth.create_client.return_value = None
            with pytest.raises(HTTPException) as exc:
                await oauth_redirect("fakeprovider", request)
            assert exc.value.status_code == 400


class TestGetMe:
    async def test_returns_user_response(self, user_factory):
        user = user_factory()
        result = await get_me(user)
        assert result.email == user.email


class TestUpdateMe:
    async def test_updates_username(self, user_factory):
        user = user_factory()
        updated = user_factory(username="newname")
        user_repo = AsyncMock()
        user_repo.update_profile = AsyncMock()
        user_repo.get_by_id = AsyncMock(return_value=updated)
        body = ProfileUpdate(username="newname")
        result = await update_me(body, user, user_repo)
        assert result.username == "newname"

    async def test_no_changes_returns_original(self, user_factory):
        user = user_factory()
        user_repo = AsyncMock()
        body = ProfileUpdate()
        result = await update_me(body, user, user_repo)
        assert result.email == user.email


class TestOAuthCallback:
    async def test_google_new_user(self, user_factory):
        user = user_factory(email="goog@x.com", username="goog", oauth_provider="google")
        user_repo = AsyncMock()
        user_repo.get_by_oauth = AsyncMock(return_value=None)
        user_repo.get_by_email = AsyncMock(return_value=None)
        user_repo.create = AsyncMock(return_value=user)
        session_repo = AsyncMock()
        session_repo.create = AsyncMock()
        request = MagicMock()
        resp = Response()

        mock_client = AsyncMock()
        mock_client.authorize_access_token = AsyncMock(return_value={
            "userinfo": {"email": "goog@x.com", "sub": "g-sub-1", "name": "Goog"},
        })

        with patch("orca_web.api.routers.auth.oauth") as mock_oauth:
            mock_oauth.create_client.return_value = mock_client
            result = await oauth_callback("google", request, resp, user_repo, session_repo)
        assert result.access_token
        user_repo.create.assert_awaited_once()

    async def test_google_existing_oauth_user(self, user_factory):
        user = user_factory(email="goog@x.com", oauth_provider="google", oauth_sub="g-sub-1")
        user_repo = AsyncMock()
        user_repo.get_by_oauth = AsyncMock(return_value=user)
        session_repo = AsyncMock()
        session_repo.create = AsyncMock()
        request = MagicMock()

        mock_client = AsyncMock()
        mock_client.authorize_access_token = AsyncMock(return_value={
            "userinfo": {"email": "goog@x.com", "sub": "g-sub-1", "name": "Goog"},
        })

        with patch("orca_web.api.routers.auth.oauth") as mock_oauth:
            mock_oauth.create_client.return_value = mock_client
            result = await oauth_callback("google", request, Response(), user_repo, session_repo)
        assert result.access_token
        user_repo.create.assert_not_awaited()

    async def test_google_link_existing_email(self, user_factory):
        """An existing user with matching email gets their OAuth fields updated."""
        user = user_factory(email="existing@x.com")
        user_repo = AsyncMock()
        user_repo.get_by_oauth = AsyncMock(return_value=None)
        user_repo.get_by_email = AsyncMock(return_value=user)
        user_repo.update_profile = AsyncMock()
        session_repo = AsyncMock()
        session_repo.create = AsyncMock()
        request = MagicMock()

        mock_client = AsyncMock()
        mock_client.authorize_access_token = AsyncMock(return_value={
            "userinfo": {"email": "existing@x.com", "sub": "g-sub-2", "name": "Existing"},
        })

        with patch("orca_web.api.routers.auth.oauth") as mock_oauth:
            mock_oauth.create_client.return_value = mock_client
            result = await oauth_callback("google", request, Response(), user_repo, session_repo)
        assert result.access_token
        user_repo.update_profile.assert_awaited_once()

    async def test_github_new_user(self, user_factory):
        user = user_factory(email="gh@x.com", username="ghuser")
        user_repo = AsyncMock()
        user_repo.get_by_oauth = AsyncMock(return_value=None)
        user_repo.get_by_email = AsyncMock(return_value=None)
        user_repo.create = AsyncMock(return_value=user)
        session_repo = AsyncMock()
        session_repo.create = AsyncMock()
        request = MagicMock()

        mock_client = AsyncMock()
        token_data = {"access_token": "gh-tok"}
        mock_client.authorize_access_token = AsyncMock(return_value=token_data)
        user_resp = MagicMock()
        user_resp.json.return_value = {"email": "gh@x.com", "id": 12345, "login": "ghuser"}
        mock_client.get = AsyncMock(return_value=user_resp)

        with patch("orca_web.api.routers.auth.oauth") as mock_oauth:
            mock_oauth.create_client.return_value = mock_client
            result = await oauth_callback("github", request, Response(), user_repo, session_repo)
        assert result.access_token
        user_repo.create.assert_awaited_once()

    async def test_github_fallback_to_primary_email(self, user_factory):
        """When GitHub user endpoint returns no email, fetches from /user/emails."""
        user = user_factory(email="primary@x.com", username="ghuser2")
        user_repo = AsyncMock()
        user_repo.get_by_oauth = AsyncMock(return_value=None)
        user_repo.get_by_email = AsyncMock(return_value=None)
        user_repo.create = AsyncMock(return_value=user)
        session_repo = AsyncMock()
        session_repo.create = AsyncMock()
        request = MagicMock()

        mock_client = AsyncMock()
        token_data = {"access_token": "gh-tok"}
        mock_client.authorize_access_token = AsyncMock(return_value=token_data)
        user_resp = MagicMock()
        user_resp.json.return_value = {"email": None, "id": 67890, "login": "ghuser2"}
        emails_resp = MagicMock()
        emails_resp.json.return_value = [
            {"email": "secondary@x.com", "primary": False},
            {"email": "primary@x.com", "primary": True},
        ]
        mock_client.get = AsyncMock(side_effect=[user_resp, emails_resp])

        with patch("orca_web.api.routers.auth.oauth") as mock_oauth:
            mock_oauth.create_client.return_value = mock_client
            result = await oauth_callback("github", request, Response(), user_repo, session_repo)
        assert result.access_token

    async def test_unsupported_provider_returns_400(self):
        mock_client = AsyncMock()
        mock_client.authorize_access_token = AsyncMock(return_value={})
        request = MagicMock()

        with patch("orca_web.api.routers.auth.oauth") as mock_oauth:
            mock_oauth.create_client.return_value = mock_client
            with pytest.raises(HTTPException) as exc:
                await oauth_callback("twitter", request, Response(), AsyncMock(), AsyncMock())
        assert exc.value.status_code == 400

    async def test_missing_email_returns_400(self):
        mock_client = AsyncMock()
        mock_client.authorize_access_token = AsyncMock(return_value={
            "userinfo": {"email": None, "sub": "g-sub-3"},
        })
        request = MagicMock()

        with patch("orca_web.api.routers.auth.oauth") as mock_oauth:
            mock_oauth.create_client.return_value = mock_client
            with pytest.raises(HTTPException) as exc:
                await oauth_callback("google", request, Response(), AsyncMock(), AsyncMock())
        assert exc.value.status_code == 400

    async def test_unconfigured_provider_callback_returns_400(self):
        request = MagicMock()
        with patch("orca_web.api.routers.auth.oauth") as mock_oauth:
            mock_oauth.create_client.return_value = None
            with pytest.raises(HTTPException) as exc:
                await oauth_callback("fakeprovider", request, Response(), AsyncMock(), AsyncMock())
        assert exc.value.status_code == 400
