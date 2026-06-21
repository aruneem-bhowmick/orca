"""Tests for orca_web.auth.jwt."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from jose import jwt as jose_jwt

from orca_web.auth.jwt import (
    create_access_token,
    create_refresh_token,
    decode_token,
    decode_token_safe,
)

SECRET = "test-jwt-secret"
ALGO = "HS256"


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch):
    """Patch jwt module settings for every test."""
    s = SimpleNamespace(
        jwt_secret_key=SECRET,
        jwt_algorithm=ALGO,
        access_token_expire_minutes=15,
        refresh_token_expire_days=7,
    )
    monkeypatch.setattr("orca_web.auth.jwt.settings", s)


class TestCreateAccessToken:
    def test_returns_decodable_jwt(self):
        token = create_access_token("user-123")
        payload = jose_jwt.decode(token, SECRET, algorithms=[ALGO])
        assert payload["sub"] == "user-123"
        assert payload["type"] == "access"
        assert "jti" in payload
        assert "exp" in payload
        assert "iat" in payload

    def test_extra_claims_merged(self):
        token = create_access_token("user-456", extra={"role": "admin"})
        payload = jose_jwt.decode(token, SECRET, algorithms=[ALGO])
        assert payload["role"] == "admin"
        assert payload["sub"] == "user-456"

    def test_expiry_matches_configured_minutes(self):
        token = create_access_token("user-789")
        payload = jose_jwt.decode(token, SECRET, algorithms=[ALGO])
        iat = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        delta = exp - iat
        assert abs(delta.total_seconds() - 15 * 60) < 2


class TestCreateRefreshToken:
    def test_returns_three_tuple(self):
        token, jti, expires_at = create_refresh_token("user-111")
        assert isinstance(token, str)
        assert isinstance(jti, str)
        assert isinstance(expires_at, datetime)

    def test_token_has_refresh_type(self):
        token, jti, _ = create_refresh_token("user-222")
        payload = jose_jwt.decode(token, SECRET, algorithms=[ALGO])
        assert payload["type"] == "refresh"
        assert payload["jti"] == jti

    def test_expiry_matches_configured_days(self):
        token, _, expires_at = create_refresh_token("user-333")
        payload = jose_jwt.decode(token, SECRET, algorithms=[ALGO])
        iat = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
        delta = expires_at - iat
        assert abs(delta.total_seconds() - 7 * 86400) < 2


class TestDecodeToken:
    def test_decodes_valid_token(self):
        token = create_access_token("user-aaa")
        payload = decode_token(token)
        assert payload["sub"] == "user-aaa"

    def test_raises_on_bad_signature(self):
        token = jose_jwt.encode(
            {"sub": "x", "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
            "wrong-secret",
            algorithm=ALGO,
        )
        with pytest.raises(Exception):
            decode_token(token)

    def test_raises_on_expired_token(self):
        token = jose_jwt.encode(
            {"sub": "x", "exp": datetime.now(timezone.utc) - timedelta(hours=1)},
            SECRET,
            algorithm=ALGO,
        )
        with pytest.raises(Exception):
            decode_token(token)

    def test_raises_on_garbage_string(self):
        with pytest.raises(Exception):
            decode_token("not.a.jwt")


class TestDecodeTokenSafe:
    def test_returns_payload_for_valid_token(self):
        token = create_access_token("user-bbb")
        result = decode_token_safe(token)
        assert result is not None
        assert result["sub"] == "user-bbb"

    def test_returns_none_for_bad_signature(self):
        token = jose_jwt.encode(
            {"sub": "x", "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
            "wrong-secret",
            algorithm=ALGO,
        )
        assert decode_token_safe(token) is None

    def test_returns_none_for_expired_token(self):
        token = jose_jwt.encode(
            {"sub": "x", "exp": datetime.now(timezone.utc) - timedelta(hours=1)},
            SECRET,
            algorithm=ALGO,
        )
        assert decode_token_safe(token) is None

    def test_returns_none_for_garbage(self):
        assert decode_token_safe("garbage") is None
