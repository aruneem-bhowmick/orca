"""Tests for orca_web.api.routers.history endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from starlette.testclient import TestClient

from orca_web.api.deps import get_current_user, get_history_repo
from orca_web.api.routers.history import (
    BookmarkCreateRequest,
    PaginatedResponse,
    _paginate,
    _serialize_activity,
    _serialize_bookmark,
    create_bookmark,
    delete_bookmark,
    global_feed,
    list_bookmarks,
    list_experiment_history,
    list_history,
    list_task_history,
    router,
)


# ── Factories ─────────────────────────────────────────────────────────────


def _make_activity(
    *,
    action="task_created",
    service="orcamind",
    resource_type="task",
    resource_id=None,
    user_id=None,
):
    """Build a MagicMock that behaves like an ActivityLog row."""
    row = MagicMock()
    row.log_id = uuid.uuid4()
    row.user_id = user_id or uuid.uuid4()
    row.action = action
    row.resource_type = resource_type
    row.resource_id = resource_id or str(uuid.uuid4())
    row.service = service
    row.details = None
    row.created_at = datetime.now(timezone.utc)
    return row


def _make_bookmark(*, user_id=None, resource_type="task", note=None):
    """Build a MagicMock that behaves like a UserBookmark row."""
    row = MagicMock()
    row.bookmark_id = uuid.uuid4()
    row.user_id = user_id or uuid.uuid4()
    row.resource_type = resource_type
    row.resource_id = str(uuid.uuid4())
    row.note = note
    row.created_at = datetime.now(timezone.utc)
    return row


# ── Serialization helpers ─────────────────────────────────────────────────


class TestSerializeActivity:
    """Verify _serialize_activity produces the expected dict structure."""

    def test_includes_all_fields(self):
        row = _make_activity(action="experiment_started", service="orcalab")
        result = _serialize_activity(row)
        assert result["action"] == "experiment_started"
        assert result["service"] == "orcalab"
        assert "log_id" in result
        assert "user_id" in result
        assert "created_at" in result

    def test_handles_none_created_at(self):
        row = _make_activity()
        row.created_at = None
        result = _serialize_activity(row)
        assert result["created_at"] is None


class TestSerializeBookmark:
    """Verify _serialize_bookmark produces the expected dict structure."""

    def test_includes_all_fields(self):
        row = _make_bookmark(note="my note")
        result = _serialize_bookmark(row)
        assert result["note"] == "my note"
        assert "bookmark_id" in result
        assert "resource_type" in result

    def test_handles_none_created_at(self):
        row = _make_bookmark()
        row.created_at = None
        result = _serialize_bookmark(row)
        assert result["created_at"] is None


class TestPaginate:
    """Verify the pagination envelope builder."""

    def test_basic_pagination(self):
        result = _paginate([{"a": 1}], total=1, page=1, per_page=20)
        assert isinstance(result, PaginatedResponse)
        assert result.total == 1
        assert result.page == 1
        assert result.per_page == 20
        assert result.pages == 1
        assert len(result.items) == 1

    def test_multiple_pages(self):
        result = _paginate([], total=42, page=2, per_page=20)
        assert result.pages == 3

    def test_zero_total_returns_one_page(self):
        result = _paginate([], total=0, page=1, per_page=20)
        assert result.pages == 1

    def test_exact_multiple(self):
        result = _paginate([], total=100, page=5, per_page=20)
        assert result.pages == 5


# ── GET /history ──────────────────────────────────────────────────────────


class TestListHistory:
    """Test the paginated user activity log endpoint."""

    async def test_returns_paginated_response(self, user_factory):
        user = user_factory()
        rows = [_make_activity(user_id=user.user_id) for _ in range(3)]
        history_repo = AsyncMock()
        history_repo.count_for_user = AsyncMock(return_value=3)
        history_repo.list_for_user = AsyncMock(return_value=rows)

        result = await list_history(
            page=1, per_page=20, user=user, history_repo=history_repo
        )
        assert isinstance(result, PaginatedResponse)
        assert result.total == 3
        assert len(result.items) == 3
        assert result.page == 1
        assert result.per_page == 20

    async def test_pagination_offset_calculation(self, user_factory):
        user = user_factory()
        history_repo = AsyncMock()
        history_repo.count_for_user = AsyncMock(return_value=50)
        history_repo.list_for_user = AsyncMock(return_value=[])

        await list_history(
            page=3, per_page=10, user=user, history_repo=history_repo
        )
        history_repo.list_for_user.assert_awaited_once_with(
            user.user_id, limit=10, offset=20
        )

    async def test_empty_history(self, user_factory):
        user = user_factory()
        history_repo = AsyncMock()
        history_repo.count_for_user = AsyncMock(return_value=0)
        history_repo.list_for_user = AsyncMock(return_value=[])

        result = await list_history(
            page=1, per_page=20, user=user, history_repo=history_repo
        )
        assert result.total == 0
        assert result.items == []
        assert result.pages == 1


# ── GET /history/tasks ────────────────────────────────────────────────────


class TestListTaskHistory:
    """Test the OrcaMind-filtered activity log endpoint."""

    async def test_filters_by_orcamind_service(self, user_factory):
        user = user_factory()
        history_repo = AsyncMock()
        history_repo.count_for_user = AsyncMock(return_value=5)
        history_repo.list_for_user = AsyncMock(
            return_value=[_make_activity(service="orcamind")]
        )

        result = await list_task_history(
            page=1, per_page=20, user=user, history_repo=history_repo
        )
        history_repo.count_for_user.assert_awaited_once_with(
            user.user_id, service="orcamind"
        )
        history_repo.list_for_user.assert_awaited_once_with(
            user.user_id, limit=20, offset=0, service="orcamind"
        )
        assert isinstance(result, PaginatedResponse)


# ── GET /history/experiments ──────────────────────────────────────────────


class TestListExperimentHistory:
    """Test the OrcaLab-filtered activity log endpoint."""

    async def test_filters_by_orcalab_service(self, user_factory):
        user = user_factory()
        history_repo = AsyncMock()
        history_repo.count_for_user = AsyncMock(return_value=2)
        history_repo.list_for_user = AsyncMock(
            return_value=[_make_activity(service="orcalab")]
        )

        result = await list_experiment_history(
            page=1, per_page=20, user=user, history_repo=history_repo
        )
        history_repo.count_for_user.assert_awaited_once_with(
            user.user_id, service="orcalab"
        )
        history_repo.list_for_user.assert_awaited_once_with(
            user.user_id, limit=20, offset=0, service="orcalab"
        )
        assert isinstance(result, PaginatedResponse)


# ── POST /bookmarks ──────────────────────────────────────────────────────


class TestCreateBookmark:
    """Test bookmark creation endpoint."""

    async def test_creates_bookmark_and_returns_data(self, user_factory):
        user = user_factory()
        bookmark_row = _make_bookmark(user_id=user.user_id, note="saved")
        history_repo = AsyncMock()
        history_repo.add_bookmark = AsyncMock(return_value=bookmark_row)

        body = BookmarkCreateRequest(
            resource_type="task",
            resource_id="abc-123",
            note="saved",
        )
        result = await create_bookmark(
            body=body, user=user, history_repo=history_repo
        )
        history_repo.add_bookmark.assert_awaited_once_with(
            user_id=user.user_id,
            resource_type="task",
            resource_id="abc-123",
            note="saved",
        )
        assert result["note"] == "saved"
        assert "bookmark_id" in result

    async def test_creates_bookmark_without_note(self, user_factory):
        user = user_factory()
        bookmark_row = _make_bookmark(user_id=user.user_id)
        history_repo = AsyncMock()
        history_repo.add_bookmark = AsyncMock(return_value=bookmark_row)

        body = BookmarkCreateRequest(
            resource_type="experiment",
            resource_id="exp-1",
        )
        result = await create_bookmark(
            body=body, user=user, history_repo=history_repo
        )
        history_repo.add_bookmark.assert_awaited_once_with(
            user_id=user.user_id,
            resource_type="experiment",
            resource_id="exp-1",
            note=None,
        )
        assert "bookmark_id" in result


# ── DELETE /bookmarks/{bookmark_id} ──────────────────────────────────────


class TestDeleteBookmark:
    """Test bookmark deletion with ownership verification."""

    async def test_deletes_own_bookmark(self, user_factory):
        user = user_factory()
        bid = uuid.uuid4()
        existing = MagicMock()
        existing.bookmark_id = bid
        existing.user_id = user.user_id

        history_repo = AsyncMock()
        history_repo.get_bookmark_by_id = AsyncMock(return_value=existing)
        history_repo.delete_bookmark = AsyncMock(return_value=True)

        await delete_bookmark(
            bookmark_id=bid, user=user, history_repo=history_repo
        )
        history_repo.delete_bookmark.assert_awaited_once_with(
            bid, user.user_id
        )

    async def test_returns_404_for_nonexistent_bookmark(self, user_factory):
        user = user_factory()
        history_repo = AsyncMock()
        history_repo.get_bookmark_by_id = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc:
            await delete_bookmark(
                bookmark_id=uuid.uuid4(),
                user=user,
                history_repo=history_repo,
            )
        assert exc.value.status_code == 404
        assert "not found" in exc.value.detail.lower()

    async def test_returns_403_for_other_users_bookmark(self, user_factory):
        user = user_factory()
        other_user_id = uuid.uuid4()
        bid = uuid.uuid4()

        existing = MagicMock()
        existing.bookmark_id = bid
        existing.user_id = other_user_id  # belongs to a different user

        history_repo = AsyncMock()
        history_repo.get_bookmark_by_id = AsyncMock(return_value=existing)

        with pytest.raises(HTTPException) as exc:
            await delete_bookmark(
                bookmark_id=bid, user=user, history_repo=history_repo
            )
        assert exc.value.status_code == 403
        assert "another user" in exc.value.detail.lower()
        # Ensure delete was never called
        history_repo.delete_bookmark.assert_not_awaited()


# ── GET /bookmarks ───────────────────────────────────────────────────────


class TestListBookmarks:
    """Test the paginated bookmark list endpoint."""

    async def test_returns_paginated_bookmarks(self, user_factory):
        user = user_factory()
        bookmarks = [_make_bookmark(user_id=user.user_id) for _ in range(2)]
        history_repo = AsyncMock()
        history_repo.count_bookmarks = AsyncMock(return_value=2)
        history_repo.list_bookmarks = AsyncMock(return_value=bookmarks)

        result = await list_bookmarks(
            page=1, per_page=20, user=user, history_repo=history_repo
        )
        assert isinstance(result, PaginatedResponse)
        assert result.total == 2
        assert len(result.items) == 2

    async def test_pagination_parameters_forwarded(self, user_factory):
        user = user_factory()
        history_repo = AsyncMock()
        history_repo.count_bookmarks = AsyncMock(return_value=30)
        history_repo.list_bookmarks = AsyncMock(return_value=[])

        await list_bookmarks(
            page=2, per_page=15, user=user, history_repo=history_repo
        )
        history_repo.list_bookmarks.assert_awaited_once_with(
            user.user_id, limit=15, offset=15
        )

    async def test_empty_bookmarks(self, user_factory):
        user = user_factory()
        history_repo = AsyncMock()
        history_repo.count_bookmarks = AsyncMock(return_value=0)
        history_repo.list_bookmarks = AsyncMock(return_value=[])

        result = await list_bookmarks(
            page=1, per_page=20, user=user, history_repo=history_repo
        )
        assert result.total == 0
        assert result.items == []


# ── GET /feed ─────────────────────────────────────────────────────────────


class TestGlobalFeed:
    """Test the global cross-user activity feed endpoint."""

    async def test_returns_paginated_feed(self, user_factory):
        user = user_factory()
        rows = [_make_activity() for _ in range(5)]
        history_repo = AsyncMock()
        history_repo.count_global_feed = AsyncMock(return_value=5)
        history_repo.list_global_feed = AsyncMock(return_value=rows)

        result = await global_feed(
            page=1, per_page=20, user=user, history_repo=history_repo
        )
        assert isinstance(result, PaginatedResponse)
        assert result.total == 5
        assert len(result.items) == 5

    async def test_pagination_offset_for_feed(self, user_factory):
        user = user_factory()
        history_repo = AsyncMock()
        history_repo.count_global_feed = AsyncMock(return_value=100)
        history_repo.list_global_feed = AsyncMock(return_value=[])

        await global_feed(
            page=4, per_page=25, user=user, history_repo=history_repo
        )
        history_repo.list_global_feed.assert_awaited_once_with(
            limit=25, offset=75
        )

    async def test_empty_feed(self, user_factory):
        user = user_factory()
        history_repo = AsyncMock()
        history_repo.count_global_feed = AsyncMock(return_value=0)
        history_repo.list_global_feed = AsyncMock(return_value=[])

        result = await global_feed(
            page=1, per_page=20, user=user, history_repo=history_repo
        )
        assert result.total == 0
        assert result.pages == 1


# ── Pagination edge cases ────────────────────────────────────────────────


class TestPaginationEdgeCases:
    """Verify pagination behaviour at boundary values."""

    async def test_page_beyond_total_returns_empty_items(self, user_factory):
        user = user_factory()
        history_repo = AsyncMock()
        history_repo.count_for_user = AsyncMock(return_value=5)
        history_repo.list_for_user = AsyncMock(return_value=[])

        result = await list_history(
            page=10, per_page=20, user=user, history_repo=history_repo
        )
        assert result.total == 5
        assert result.items == []
        assert result.page == 10

    async def test_per_page_one_calculates_pages_correctly(self, user_factory):
        user = user_factory()
        history_repo = AsyncMock()
        history_repo.count_for_user = AsyncMock(return_value=5)
        history_repo.list_for_user = AsyncMock(
            return_value=[_make_activity()]
        )

        result = await list_history(
            page=1, per_page=1, user=user, history_repo=history_repo
        )
        assert result.pages == 5

    async def test_max_per_page(self, user_factory):
        user = user_factory()
        history_repo = AsyncMock()
        history_repo.count_for_user = AsyncMock(return_value=200)
        history_repo.list_for_user = AsyncMock(return_value=[])

        result = await list_history(
            page=1, per_page=100, user=user, history_repo=history_repo
        )
        assert result.per_page == 100
        assert result.pages == 2


# ── HTTP-level validation (TestClient) ───────────────────────────────────


def _build_test_client(user_factory):
    """Build a TestClient with overridden auth and repository dependencies.

    The history repo mock returns empty results for all queries so
    that the tests can focus on FastAPI's query-parameter validation
    and default binding without needing real data.
    """
    app = FastAPI()
    app.include_router(router)

    mock_user = user_factory()
    mock_repo = AsyncMock()
    mock_repo.count_for_user = AsyncMock(return_value=0)
    mock_repo.list_for_user = AsyncMock(return_value=[])

    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_history_repo] = lambda: mock_repo

    return TestClient(app), mock_repo


class TestQueryParameterValidation:
    """Verify that FastAPI enforces Query constraints at the HTTP level.

    Direct handler calls bypass FastAPI's parameter validation, so
    these tests use TestClient to exercise the full request path.
    """

    def test_per_page_zero_returns_422(self, user_factory):
        client, _ = _build_test_client(user_factory)
        resp = client.get("/history?per_page=0")
        assert resp.status_code == 422

    def test_per_page_over_max_returns_422(self, user_factory):
        client, _ = _build_test_client(user_factory)
        resp = client.get("/history?per_page=101")
        assert resp.status_code == 422

    def test_page_zero_returns_422(self, user_factory):
        client, _ = _build_test_client(user_factory)
        resp = client.get("/history?page=0")
        assert resp.status_code == 422

    def test_page_negative_returns_422(self, user_factory):
        client, _ = _build_test_client(user_factory)
        resp = client.get("/history?page=-1")
        assert resp.status_code == 422

    def test_defaults_to_page_1_per_page_20(self, user_factory):
        client, mock_repo = _build_test_client(user_factory)
        resp = client.get("/history")
        assert resp.status_code == 200
        body = resp.json()
        assert body["page"] == 1
        assert body["per_page"] == 20
