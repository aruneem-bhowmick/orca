"""Authentication endpoints: register, login, refresh, logout, OAuth, profile."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr

from orca_web.auth.jwt import create_access_token, create_refresh_token, decode_token_safe
from orca_web.auth.oauth import oauth
from orca_web.auth.password import hash_password, verify_password
from orca_web.config import settings
from orca_web.api.deps import get_current_user, get_session_repo, get_user_repo
from orca_web.models.user import User
from orca_web.repository.session_repo import SessionRepository
from orca_web.repository.user_repo import UserRepository

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Request / response schemas ────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    user_id: UUID
    email: str
    username: str
    role: str
    preferences: dict[str, Any] | None = None

    model_config = {"from_attributes": True}


class ProfileUpdate(BaseModel):
    username: str | None = None
    preferences: dict[str, Any] | None = None


# ── Helpers ───────────────────────────────────────────────────────────────

def _set_refresh_cookie(response: Response, token: str) -> None:
    """Set an httponly refresh_token cookie scoped to ``/api/v1/auth``."""
    response.set_cookie(
        key="refresh_token",
        value=token,
        httponly=True,
        secure=False,  # set True in production behind HTTPS
        samesite="lax",
        max_age=settings.refresh_token_expire_days * 86400,
        path="/api/v1/auth",
    )


def _clear_refresh_cookie(response: Response) -> None:
    """Delete the refresh_token cookie."""
    response.delete_cookie(key="refresh_token", path="/api/v1/auth")


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    response: Response,
    user_repo: UserRepository = Depends(get_user_repo),
    session_repo: SessionRepository = Depends(get_session_repo),
) -> TokenResponse:
    """Register a new user with email and password.  Returns an access token."""
    if await user_repo.get_by_email(body.email):
        raise HTTPException(status_code=409, detail="Email already registered")
    if await user_repo.get_by_username(body.username):
        raise HTTPException(status_code=409, detail="Username already taken")

    user = await user_repo.create(
        email=body.email,
        username=body.username,
        password_hash=hash_password(body.password),
    )

    access = create_access_token(str(user.user_id))
    refresh, jti, expires_at = create_refresh_token(str(user.user_id))
    await session_repo.create(user_id=user.user_id, jti=jti, expires_at=expires_at)
    _set_refresh_cookie(response, refresh)
    return TokenResponse(access_token=access)


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    response: Response,
    user_repo: UserRepository = Depends(get_user_repo),
    session_repo: SessionRepository = Depends(get_session_repo),
) -> TokenResponse:
    """Authenticate by email and password.  Returns an access token."""
    user = await user_repo.get_by_email(body.email)
    if user is None or user.password_hash is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access = create_access_token(str(user.user_id))
    refresh, jti, expires_at = create_refresh_token(str(user.user_id))
    await session_repo.create(user_id=user.user_id, jti=jti, expires_at=expires_at)
    _set_refresh_cookie(response, refresh)
    return TokenResponse(access_token=access)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: Request,
    response: Response,
    user_repo: UserRepository = Depends(get_user_repo),
    session_repo: SessionRepository = Depends(get_session_repo),
) -> TokenResponse:
    """Rotate the refresh token and return a new access token."""
    token = request.cookies.get("refresh_token")
    if not token:
        raise HTTPException(status_code=401, detail="No refresh token")

    payload = decode_token_safe(token)
    if payload is None or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    jti = payload.get("jti")
    if not jti:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    db_session = await session_repo.get_by_jti(jti)
    if db_session is None or db_session.revoked:
        raise HTTPException(status_code=401, detail="Refresh token revoked")

    user_id = payload["sub"]
    user = await user_repo.get_by_id(user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found")

    # Rotate: revoke old, issue new
    await session_repo.revoke(jti)
    access = create_access_token(str(user.user_id))
    new_refresh, new_jti, expires_at = create_refresh_token(str(user.user_id))
    await session_repo.create(user_id=user.user_id, jti=new_jti, expires_at=expires_at)
    _set_refresh_cookie(response, new_refresh)
    return TokenResponse(access_token=access)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    response: Response,
    session_repo: SessionRepository = Depends(get_session_repo),
) -> None:
    """Revoke the current refresh token and clear the cookie."""
    token = request.cookies.get("refresh_token")
    if token:
        payload = decode_token_safe(token)
        if payload and (jti := payload.get("jti")):
            await session_repo.revoke(jti)
    _clear_refresh_cookie(response)


# ── OAuth ─────────────────────────────────────────────────────────────────

@router.get("/oauth/{provider}")
async def oauth_redirect(provider: str, request: Request) -> Any:
    """Redirect the user to the OAuth provider's authorization page."""
    client = oauth.create_client(provider)
    if client is None:
        raise HTTPException(status_code=400, detail=f"OAuth provider '{provider}' not configured")
    redirect_uri = f"{settings.frontend_url}/oauth/callback?provider={provider}"
    return await client.authorize_redirect(request, redirect_uri)


@router.get("/oauth/{provider}/callback", response_model=TokenResponse)
async def oauth_callback(
    provider: str,
    request: Request,
    response: Response,
    user_repo: UserRepository = Depends(get_user_repo),
    session_repo: SessionRepository = Depends(get_session_repo),
) -> TokenResponse:
    """Handle the OAuth callback: exchange code for token, upsert user, return access token."""
    client = oauth.create_client(provider)
    if client is None:
        raise HTTPException(status_code=400, detail=f"OAuth provider '{provider}' not configured")

    token_data = await client.authorize_access_token(request)

    if provider == "google":
        user_info = token_data.get("userinfo", {})
        email = user_info.get("email")
        sub = user_info.get("sub")
        name = user_info.get("name", email.split("@")[0] if email else "user")
    elif provider == "github":
        resp = await client.get("user", token=token_data)
        user_info = resp.json()
        email = user_info.get("email")
        sub = str(user_info.get("id"))
        name = user_info.get("login", "user")
        if not email:
            emails_resp = await client.get("user/emails", token=token_data)
            for e in emails_resp.json():
                if e.get("primary"):
                    email = e["email"]
                    break
    else:
        raise HTTPException(status_code=400, detail="Unsupported provider")

    if not email or not sub:
        raise HTTPException(status_code=400, detail="Could not retrieve email from provider")

    user = await user_repo.get_by_oauth(provider, sub)
    if user is None:
        user = await user_repo.get_by_email(email)
        if user is None:
            user = await user_repo.create(
                email=email,
                username=name,
                oauth_provider=provider,
                oauth_sub=sub,
            )
        else:
            await user_repo.update_profile(user.user_id, oauth_provider=provider, oauth_sub=sub)

    access = create_access_token(str(user.user_id))
    refresh, jti, expires_at = create_refresh_token(str(user.user_id))
    await session_repo.create(user_id=user.user_id, jti=jti, expires_at=expires_at)
    _set_refresh_cookie(response, refresh)
    return TokenResponse(access_token=access)


# ── Profile ───────────────────────────────────────────────────────────────

@router.get("/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_current_user)) -> UserResponse:
    """Return the profile of the currently authenticated user."""
    return UserResponse.model_validate(user)


@router.patch("/me", response_model=UserResponse)
async def update_me(
    body: ProfileUpdate,
    user: User = Depends(get_current_user),
    user_repo: UserRepository = Depends(get_user_repo),
) -> UserResponse:
    """Update the current user's username and/or preferences."""
    updates = body.model_dump(exclude_unset=True)
    if updates:
        await user_repo.update_profile(user.user_id, **updates)
        updated = await user_repo.get_by_id(user.user_id)
        if updated:
            return UserResponse.model_validate(updated)
    return UserResponse.model_validate(user)
