"""User profile CRUD (admin-level operations)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from orca_web.api.deps import get_current_user, get_user_repo
from orca_web.api.routers.auth import UserResponse
from orca_web.models.user import User
from orca_web.repository.user_repo import UserRepository

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: UUID,
    current_user: User = Depends(get_current_user),
    user_repo: UserRepository = Depends(get_user_repo),
) -> UserResponse:
    if current_user.role != "admin" and current_user.user_id != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    user = await user_repo.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse.model_validate(user)
