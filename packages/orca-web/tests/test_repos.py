"""Tests for orca_web.repository (user, session, history)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from orca_web.repository.user_repo import UserRepository
from orca_web.repository.session_repo import SessionRepository
from orca_web.repository.history_repo import HistoryRepository


# ---------------------------------------------------------------------------
# UserRepository
# ---------------------------------------------------------------------------

class TestUserRepository:
    async def test_create_adds_and_flushes(self, mock_async_session):
        repo = UserRepository(mock_async_session)
        user = await repo.create(email="a@b.com", username="ab")
        mock_async_session.add.assert_called_once()
        mock_async_session.flush.assert_awaited_once()
        assert user.email == "a@b.com"

    async def test_create_with_password(self, mock_async_session):
        repo = UserRepository(mock_async_session)
        user = await repo.create(
            email="c@d.com", username="cd", password_hash="hashed"
        )
        assert user.password_hash == "hashed"

    async def test_create_with_oauth(self, mock_async_session):
        repo = UserRepository(mock_async_session)
        user = await repo.create(
            email="e@f.com",
            username="ef",
            oauth_provider="google",
            oauth_sub="sub-123",
        )
        assert user.oauth_provider == "google"

    async def test_get_by_id_returns_none_when_missing(self, mock_async_session):
        repo = UserRepository(mock_async_session)
        result = await repo.get_by_id(uuid.uuid4())
        assert result is None

    async def test_get_by_email_calls_execute(self, mock_async_session):
        repo = UserRepository(mock_async_session)
        await repo.get_by_email("a@b.com")
        mock_async_session.execute.assert_awaited()

    async def test_get_by_username_calls_execute(self, mock_async_session):
        repo = UserRepository(mock_async_session)
        await repo.get_by_username("ab")
        mock_async_session.execute.assert_awaited()

    async def test_get_by_oauth_calls_execute(self, mock_async_session):
        repo = UserRepository(mock_async_session)
        await repo.get_by_oauth("github", "sub-1")
        mock_async_session.execute.assert_awaited()

    async def test_update_profile_executes_and_flushes(self, mock_async_session):
        repo = UserRepository(mock_async_session)
        await repo.update_profile(uuid.uuid4(), username="new")
        mock_async_session.execute.assert_awaited()
        mock_async_session.flush.assert_awaited()


# ---------------------------------------------------------------------------
# SessionRepository
# ---------------------------------------------------------------------------

class TestSessionRepository:
    async def test_create_adds_and_flushes(self, mock_async_session):
        repo = SessionRepository(mock_async_session)
        row = await repo.create(
            user_id=uuid.uuid4(),
            jti="jti-1",
            expires_at=datetime.now(timezone.utc),
        )
        mock_async_session.add.assert_called_once()
        mock_async_session.flush.assert_awaited_once()
        assert row.jti == "jti-1"

    async def test_get_by_jti_returns_none_when_missing(self, mock_async_session):
        repo = SessionRepository(mock_async_session)
        result = await repo.get_by_jti("nonexistent")
        assert result is None

    async def test_revoke_executes_update(self, mock_async_session):
        repo = SessionRepository(mock_async_session)
        await repo.revoke("jti-2")
        mock_async_session.execute.assert_awaited()
        mock_async_session.flush.assert_awaited()

    async def test_revoke_all_for_user(self, mock_async_session):
        repo = SessionRepository(mock_async_session)
        await repo.revoke_all_for_user(uuid.uuid4())
        mock_async_session.execute.assert_awaited()
        mock_async_session.flush.assert_awaited()


# ---------------------------------------------------------------------------
# HistoryRepository
# ---------------------------------------------------------------------------

class TestHistoryRepository:
    async def test_log_activity_adds_and_flushes(self, mock_async_session):
        repo = HistoryRepository(mock_async_session)
        row = await repo.log_activity(
            user_id=uuid.uuid4(), action="view", service="orcamind"
        )
        mock_async_session.add.assert_called_once()
        mock_async_session.flush.assert_awaited_once()
        assert row.action == "view"

    async def test_list_for_user_calls_execute(self, mock_async_session):
        repo = HistoryRepository(mock_async_session)
        result = await repo.list_for_user(uuid.uuid4())
        assert result == []
        mock_async_session.execute.assert_awaited()

    async def test_list_for_user_with_filters(self, mock_async_session):
        repo = HistoryRepository(mock_async_session)
        await repo.list_for_user(
            uuid.uuid4(), service="orcalab", resource_type="experiment"
        )
        mock_async_session.execute.assert_awaited()

    async def test_list_global_feed(self, mock_async_session):
        repo = HistoryRepository(mock_async_session)
        result = await repo.list_global_feed(limit=10)
        assert result == []

    async def test_add_bookmark(self, mock_async_session):
        repo = HistoryRepository(mock_async_session)
        row = await repo.add_bookmark(
            user_id=uuid.uuid4(),
            resource_type="task",
            resource_id="t-1",
            note="important",
        )
        assert row.resource_type == "task"
        mock_async_session.add.assert_called()
        mock_async_session.flush.assert_awaited()

    async def test_delete_bookmark_returns_false_when_not_found(
        self, mock_async_session
    ):
        mock_async_session.execute.return_value.rowcount = 0
        repo = HistoryRepository(mock_async_session)
        result = await repo.delete_bookmark(uuid.uuid4(), uuid.uuid4())
        assert result is False

    async def test_delete_bookmark_returns_true_when_found(
        self, mock_async_session
    ):
        mock_async_session.execute.return_value.rowcount = 1
        repo = HistoryRepository(mock_async_session)
        result = await repo.delete_bookmark(uuid.uuid4(), uuid.uuid4())
        assert result is True

    async def test_list_bookmarks(self, mock_async_session):
        repo = HistoryRepository(mock_async_session)
        result = await repo.list_bookmarks(uuid.uuid4())
        assert result == []

    async def test_count_for_user_returns_scalar(self, mock_async_session):
        mock_async_session.execute.return_value.scalar_one.return_value = 5
        repo = HistoryRepository(mock_async_session)
        result = await repo.count_for_user(uuid.uuid4())
        assert result == 5
        mock_async_session.execute.assert_awaited()

    async def test_count_for_user_with_service_filter(self, mock_async_session):
        mock_async_session.execute.return_value.scalar_one.return_value = 3
        repo = HistoryRepository(mock_async_session)
        result = await repo.count_for_user(uuid.uuid4(), service="orcamind")
        assert result == 3

    async def test_count_for_user_with_resource_type_filter(self, mock_async_session):
        mock_async_session.execute.return_value.scalar_one.return_value = 2
        repo = HistoryRepository(mock_async_session)
        result = await repo.count_for_user(
            uuid.uuid4(), resource_type="task"
        )
        assert result == 2

    async def test_count_for_user_with_both_filters(self, mock_async_session):
        mock_async_session.execute.return_value.scalar_one.return_value = 1
        repo = HistoryRepository(mock_async_session)
        result = await repo.count_for_user(
            uuid.uuid4(), service="orcamind", resource_type="task"
        )
        assert result == 1

    async def test_count_global_feed(self, mock_async_session):
        mock_async_session.execute.return_value.scalar_one.return_value = 42
        repo = HistoryRepository(mock_async_session)
        result = await repo.count_global_feed()
        assert result == 42

    async def test_get_bookmark_by_id_returns_none_when_missing(
        self, mock_async_session
    ):
        repo = HistoryRepository(mock_async_session)
        result = await repo.get_bookmark_by_id(uuid.uuid4())
        assert result is None

    async def test_get_bookmark_by_id_returns_bookmark(self, mock_async_session):
        bookmark = MagicMock()
        bookmark.bookmark_id = uuid.uuid4()
        mock_async_session.execute.return_value.scalar_one_or_none.return_value = bookmark
        repo = HistoryRepository(mock_async_session)
        result = await repo.get_bookmark_by_id(bookmark.bookmark_id)
        assert result is bookmark

    async def test_count_bookmarks(self, mock_async_session):
        mock_async_session.execute.return_value.scalar_one.return_value = 7
        repo = HistoryRepository(mock_async_session)
        result = await repo.count_bookmarks(uuid.uuid4())
        assert result == 7
