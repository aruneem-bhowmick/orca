"""Tests for orca_web.api.routers.users endpoints."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from orca_web.api.routers.users import get_user


class TestGetUser:
    async def test_admin_can_access_any_user(self, user_factory):
        target_id = uuid.uuid4()
        admin = user_factory(role="admin")
        target = user_factory(user_id=target_id, email="bob@x.com", username="bob")
        user_repo = AsyncMock()
        user_repo.get_by_id = AsyncMock(return_value=target)

        result = await get_user(target_id, current_user=admin, user_repo=user_repo)
        assert result.email == "bob@x.com"
        user_repo.get_by_id.assert_awaited_once_with(target_id)

    async def test_user_can_access_own_profile(self, user_factory):
        uid = uuid.uuid4()
        user = user_factory(user_id=uid, email="self@x.com", username="self")
        user_repo = AsyncMock()
        user_repo.get_by_id = AsyncMock(return_value=user)

        result = await get_user(uid, current_user=user, user_repo=user_repo)
        assert result.email == "self@x.com"

    async def test_non_admin_cannot_access_other_user(self, user_factory):
        caller = user_factory(role="user")
        other_id = uuid.uuid4()
        user_repo = AsyncMock()

        with pytest.raises(HTTPException) as exc:
            await get_user(other_id, current_user=caller, user_repo=user_repo)
        assert exc.value.status_code == 403

    async def test_returns_404_when_user_not_found(self, user_factory):
        uid = uuid.uuid4()
        admin = user_factory(role="admin")
        user_repo = AsyncMock()
        user_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc:
            await get_user(uid, current_user=admin, user_repo=user_repo)
        assert exc.value.status_code == 404
