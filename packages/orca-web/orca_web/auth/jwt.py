"""JWT token creation and validation."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt

from orca_web.config import settings


def create_access_token(user_id: str, extra: dict[str, Any] | None = None) -> str:
    """Create a short-lived access token."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_expire_minutes),
        "jti": str(uuid.uuid4()),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(user_id: str) -> tuple[str, str, datetime]:
    """Create a long-lived refresh token.

    Returns ``(encoded_token, jti, expires_at)``.
    """
    now = datetime.now(timezone.utc)
    jti = str(uuid.uuid4())
    expires_at = now + timedelta(days=settings.refresh_token_expire_days)
    payload = {
        "sub": user_id,
        "type": "refresh",
        "iat": now,
        "exp": expires_at,
        "jti": jti,
    }
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return token, jti, expires_at


def decode_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT, raising ``JWTError`` on failure."""
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])


def decode_token_safe(token: str) -> dict[str, Any] | None:
    """Decode a JWT, returning ``None`` on any validation error."""
    try:
        return decode_token(token)
    except JWTError:
        return None
