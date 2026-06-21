"""Tests for orca_web.models.user ORM model definitions."""

import uuid
from datetime import datetime, timezone

from orca_web.models.user import ActivityLog, User, UserBookmark, UserSession


class TestUserModel:
    def test_tablename(self):
        assert User.__tablename__ == "users"

    def test_instantiation_with_defaults(self):
        uid = uuid.uuid4()
        u = User(user_id=uid, email="a@b.com", username="ab")
        assert u.user_id == uid
        assert u.email == "a@b.com"
        assert u.username == "ab"
        assert u.password_hash is None
        assert u.oauth_provider is None

    def test_relationships_declared(self):
        rels = {r.key for r in User.__mapper__.relationships}
        assert "sessions" in rels
        assert "activities" in rels
        assert "bookmarks" in rels


class TestUserSessionModel:
    def test_tablename(self):
        assert UserSession.__tablename__ == "user_sessions"

    def test_instantiation(self):
        sid = uuid.uuid4()
        uid = uuid.uuid4()
        s = UserSession(
            session_id=sid,
            user_id=uid,
            jti="jti-abc",
            expires_at=datetime.now(timezone.utc),
        )
        assert s.session_id == sid
        assert s.jti == "jti-abc"
        assert s.revoked is None or s.revoked is False  # default


class TestActivityLogModel:
    def test_tablename(self):
        assert ActivityLog.__tablename__ == "activity_log"

    def test_instantiation(self):
        lid = uuid.uuid4()
        uid = uuid.uuid4()
        a = ActivityLog(
            log_id=lid,
            user_id=uid,
            action="view",
            resource_type="task",
            resource_id="t-1",
            service="orcamind",
            details={"query": "test"},
        )
        assert a.action == "view"
        assert a.details == {"query": "test"}


class TestUserBookmarkModel:
    def test_tablename(self):
        assert UserBookmark.__tablename__ == "user_bookmarks"

    def test_instantiation(self):
        bid = uuid.uuid4()
        uid = uuid.uuid4()
        b = UserBookmark(
            bookmark_id=bid,
            user_id=uid,
            resource_type="experiment",
            resource_id="e-1",
            note="interesting",
        )
        assert b.resource_type == "experiment"
        assert b.note == "interesting"
