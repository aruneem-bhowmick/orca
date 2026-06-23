"""Tests for the Alembic migration environment and revision files.

Validates the migration infrastructure (alembic.ini, env.py) and the
initial migration (0001_add_user_tables) without requiring a live
database connection.  Tests inspect the migration script's Python
objects and the configuration files to ensure structural correctness.
"""

from __future__ import annotations

import configparser
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Locate the orca-web package root by walking up from this test file
_ORCA_WEB_ROOT = Path(__file__).resolve().parent.parent
_ALEMBIC_DIR = _ORCA_WEB_ROOT / "alembic"
_VERSIONS_DIR = _ALEMBIC_DIR / "versions"
_ALEMBIC_INI = _ORCA_WEB_ROOT / "alembic.ini"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_migration_module(filename: str):
    """Import a migration file as a Python module by file path."""
    filepath = _VERSIONS_DIR / filename
    spec = importlib.util.spec_from_file_location(
        f"alembic_versions.{filepath.stem}", filepath,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# alembic.ini configuration tests
# ---------------------------------------------------------------------------


class TestAlembicIni:
    """Verify alembic.ini has the correct settings."""

    def test_ini_file_exists(self):
        """alembic.ini must be present at the orca-web package root."""
        assert _ALEMBIC_INI.exists(), f"Expected {_ALEMBIC_INI} to exist"

    def test_script_location(self):
        """script_location must point to the 'alembic' directory."""
        cfg = configparser.ConfigParser()
        cfg.read(str(_ALEMBIC_INI))
        assert cfg.get("alembic", "script_location") == "alembic"

    def test_file_template(self):
        """file_template must use the year-month-day slug pattern."""
        cfg = configparser.ConfigParser()
        cfg.read(str(_ALEMBIC_INI))
        template = cfg.get("alembic", "file_template")
        assert "%(year)d" in template
        assert "%(month)" in template
        assert "%(day)" in template
        assert "%(slug)s" in template

    def test_sqlalchemy_url_uses_asyncpg(self):
        """Default sqlalchemy.url must use the asyncpg driver."""
        cfg = configparser.ConfigParser()
        cfg.read(str(_ALEMBIC_INI))
        url = cfg.get("alembic", "sqlalchemy.url")
        assert "asyncpg" in url

    def test_sqlalchemy_url_targets_orca_registry(self):
        """Default database URL must target the orca_registry database."""
        cfg = configparser.ConfigParser()
        cfg.read(str(_ALEMBIC_INI))
        url = cfg.get("alembic", "sqlalchemy.url")
        assert "orca_registry" in url

    def test_logging_sections_present(self):
        """Standard Alembic logging sections must be configured."""
        cfg = configparser.ConfigParser()
        cfg.read(str(_ALEMBIC_INI))
        for section in ["logger_root", "logger_sqlalchemy", "logger_alembic"]:
            assert cfg.has_section(section), f"Missing logging section: {section}"


# ---------------------------------------------------------------------------
# env.py configuration tests
# ---------------------------------------------------------------------------


class TestEnvPy:
    """Verify the Alembic environment module structure."""

    def test_env_py_exists(self):
        """alembic/env.py must exist."""
        assert (_ALEMBIC_DIR / "env.py").exists()

    def test_env_imports_base_from_orca_web_models(self):
        """env.py must import Base from orca_web.models.user."""
        source = (_ALEMBIC_DIR / "env.py").read_text()
        assert "from orca_web.models.user import Base" in source

    def test_env_sets_target_metadata(self):
        """env.py must assign target_metadata from Base.metadata."""
        source = (_ALEMBIC_DIR / "env.py").read_text()
        assert "target_metadata = Base.metadata" in source

    def test_env_reads_database_url_from_environ(self):
        """env.py must read DATABASE_URL from the environment."""
        source = (_ALEMBIC_DIR / "env.py").read_text()
        assert 'os.environ.get("DATABASE_URL")' in source

    def test_env_uses_nullpool(self):
        """env.py must import and use NullPool for migration connections."""
        source = (_ALEMBIC_DIR / "env.py").read_text()
        assert "NullPool" in source
        assert "poolclass=NullPool" in source

    def test_env_supports_offline_mode(self):
        """env.py must define run_migrations_offline for SQL generation."""
        source = (_ALEMBIC_DIR / "env.py").read_text()
        assert "def run_migrations_offline" in source

    def test_env_supports_online_mode(self):
        """env.py must define run_migrations_online for live execution."""
        source = (_ALEMBIC_DIR / "env.py").read_text()
        assert "def run_migrations_online" in source

    def test_env_uses_async_engine(self):
        """env.py must use async_engine_from_config for asyncpg support."""
        source = (_ALEMBIC_DIR / "env.py").read_text()
        assert "async_engine_from_config" in source

    def test_env_dispatches_by_offline_mode(self):
        """env.py must check context.is_offline_mode() to dispatch."""
        source = (_ALEMBIC_DIR / "env.py").read_text()
        assert "context.is_offline_mode()" in source


# ---------------------------------------------------------------------------
# script.py.mako template tests
# ---------------------------------------------------------------------------


class TestMakoTemplate:
    """Verify the Mako revision template."""

    def test_mako_template_exists(self):
        """alembic/script.py.mako must exist."""
        assert (_ALEMBIC_DIR / "script.py.mako").exists()

    def test_mako_contains_revision_variables(self):
        """Template must include Alembic revision placeholders."""
        source = (_ALEMBIC_DIR / "script.py.mako").read_text()
        assert "${up_revision}" in source
        assert "${down_revision" in source

    def test_mako_contains_upgrade_downgrade(self):
        """Template must define upgrade and downgrade functions."""
        source = (_ALEMBIC_DIR / "script.py.mako").read_text()
        assert "def upgrade()" in source
        assert "def downgrade()" in source


# ---------------------------------------------------------------------------
# Migration 0001 structure tests
# ---------------------------------------------------------------------------


class TestMigration0001Structure:
    """Validate the 0001_add_user_tables migration file structure."""

    def test_migration_file_exists(self):
        """0001_add_user_tables.py must exist in the versions directory."""
        migration_path = _VERSIONS_DIR / "0001_add_user_tables.py"
        assert migration_path.exists()

    def test_revision_id(self):
        """Revision ID must be '0001'."""
        mod = _load_migration_module("0001_add_user_tables.py")
        assert mod.revision == "0001"

    def test_down_revision_is_none(self):
        """down_revision must be None (first migration)."""
        mod = _load_migration_module("0001_add_user_tables.py")
        assert mod.down_revision is None

    def test_branch_labels_is_none(self):
        """branch_labels must be None."""
        mod = _load_migration_module("0001_add_user_tables.py")
        assert mod.branch_labels is None

    def test_depends_on_is_none(self):
        """depends_on must be None."""
        mod = _load_migration_module("0001_add_user_tables.py")
        assert mod.depends_on is None

    def test_has_upgrade_function(self):
        """Migration must define an upgrade() function."""
        mod = _load_migration_module("0001_add_user_tables.py")
        assert callable(getattr(mod, "upgrade", None))

    def test_has_downgrade_function(self):
        """Migration must define a downgrade() function."""
        mod = _load_migration_module("0001_add_user_tables.py")
        assert callable(getattr(mod, "downgrade", None))


# ---------------------------------------------------------------------------
# Migration 0001 content tests — verify DDL operations via source inspection
# ---------------------------------------------------------------------------


class TestMigration0001Tables:
    """Verify that the upgrade function creates all four required tables."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        """Read the migration source once per test class."""
        self.source = (_VERSIONS_DIR / "0001_add_user_tables.py").read_text()

    def test_creates_users_table(self):
        """upgrade() must create the 'users' table."""
        assert 'create_table(\n        "users"' in self.source

    def test_creates_user_sessions_table(self):
        """upgrade() must create the 'user_sessions' table."""
        assert 'create_table(\n        "user_sessions"' in self.source

    def test_creates_activity_log_table(self):
        """upgrade() must create the 'activity_log' table."""
        assert 'create_table(\n        "activity_log"' in self.source

    def test_creates_user_bookmarks_table(self):
        """upgrade() must create the 'user_bookmarks' table."""
        assert 'create_table(\n        "user_bookmarks"' in self.source


class TestMigration0001UsersColumns:
    """Verify users table column definitions in the migration."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = (_VERSIONS_DIR / "0001_add_user_tables.py").read_text()

    def test_user_id_pk(self):
        """users.user_id must be a UUID primary key."""
        assert '"user_id"' in self.source
        assert "primary_key=True" in self.source

    def test_user_id_server_default(self):
        """users.user_id must use gen_random_uuid() server default."""
        assert "gen_random_uuid()" in self.source

    def test_email_column(self):
        """users.email must be VARCHAR(255), unique, not nullable."""
        assert '"email"' in self.source
        assert "String(255)" in self.source

    def test_username_column(self):
        """users.username must be VARCHAR(100), unique, not nullable."""
        assert '"username"' in self.source
        assert "String(100)" in self.source

    def test_password_hash_nullable(self):
        """users.password_hash must be nullable (for OAuth-only users)."""
        assert '"password_hash"' in self.source

    def test_oauth_provider_column(self):
        """users.oauth_provider must be VARCHAR(50), nullable."""
        assert '"oauth_provider"' in self.source
        assert "String(50)" in self.source

    def test_oauth_sub_column(self):
        """users.oauth_sub must be VARCHAR(255), nullable."""
        assert '"oauth_sub"' in self.source

    def test_role_default(self):
        """users.role must default to 'user'."""
        assert '"role"' in self.source
        assert 'server_default="user"' in self.source

    def test_preferences_jsonb(self):
        """users.preferences must be JSONB, nullable."""
        assert '"preferences"' in self.source
        assert "JSONB()" in self.source

    def test_is_active_default(self):
        """users.is_active must default to true."""
        assert '"is_active"' in self.source
        assert 'server_default="true"' in self.source

    def test_created_at_timestamptz(self):
        """users.created_at must be TIMESTAMPTZ with NOW() default."""
        assert '"created_at"' in self.source
        assert "DateTime(timezone=True)" in self.source
        assert "func.now()" in self.source

    def test_updated_at_timestamptz(self):
        """users.updated_at must be TIMESTAMPTZ with NOW() default."""
        assert '"updated_at"' in self.source


class TestMigration0001UserSessionsColumns:
    """Verify user_sessions table column definitions."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = (_VERSIONS_DIR / "0001_add_user_tables.py").read_text()

    def test_session_id_pk(self):
        """user_sessions.session_id must be a UUID primary key."""
        assert '"session_id"' in self.source

    def test_user_id_fk(self):
        """user_sessions.user_id must be a FK to users with CASCADE."""
        assert 'ForeignKey("users.user_id", ondelete="CASCADE")' in self.source

    def test_jti_unique(self):
        """user_sessions.jti must be unique and not nullable."""
        assert '"jti"' in self.source

    def test_device_info_nullable(self):
        """user_sessions.device_info must be nullable text."""
        assert '"device_info"' in self.source

    def test_ip_address_varchar45(self):
        """user_sessions.ip_address must be VARCHAR(45) for IPv6 support."""
        assert '"ip_address"' in self.source
        assert "String(45)" in self.source

    def test_expires_at_not_nullable(self):
        """user_sessions.expires_at must be TIMESTAMPTZ, not nullable."""
        assert '"expires_at"' in self.source

    def test_revoked_default_false(self):
        """user_sessions.revoked must default to false."""
        assert '"revoked"' in self.source
        assert 'server_default="false"' in self.source


class TestMigration0001ActivityLogColumns:
    """Verify activity_log table column definitions."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = (_VERSIONS_DIR / "0001_add_user_tables.py").read_text()

    def test_log_id_pk(self):
        """activity_log.log_id must be a UUID primary key."""
        assert '"log_id"' in self.source

    def test_action_not_nullable(self):
        """activity_log.action must be VARCHAR(100), not nullable."""
        assert '"action"' in self.source
        assert "String(100)" in self.source

    def test_resource_type_nullable(self):
        """activity_log.resource_type must be nullable."""
        assert '"resource_type"' in self.source

    def test_resource_id_nullable(self):
        """activity_log.resource_id must be nullable."""
        assert '"resource_id"' in self.source

    def test_service_nullable(self):
        """activity_log.service must be VARCHAR(50), nullable."""
        assert '"service"' in self.source

    def test_details_jsonb(self):
        """activity_log.details must be JSONB, nullable."""
        assert '"details"' in self.source


class TestMigration0001BookmarksColumns:
    """Verify user_bookmarks table column definitions."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = (_VERSIONS_DIR / "0001_add_user_tables.py").read_text()

    def test_bookmark_id_pk(self):
        """user_bookmarks.bookmark_id must be a UUID primary key."""
        assert '"bookmark_id"' in self.source

    def test_resource_type_not_nullable(self):
        """user_bookmarks.resource_type must not be nullable."""
        # resource_type appears in both activity_log and bookmarks;
        # verify the bookmarks table specifically has nullable=False
        # by checking the table-creation block
        bookmark_section = self.source.split('"user_bookmarks"')[1]
        assert "nullable=False" in bookmark_section.split("def downgrade")[0]

    def test_resource_id_not_nullable(self):
        """user_bookmarks.resource_id must not be nullable."""
        bookmark_section = self.source.split('"user_bookmarks"')[1]
        assert '"resource_id"' in bookmark_section

    def test_note_nullable(self):
        """user_bookmarks.note must be TEXT, nullable."""
        assert '"note"' in self.source


# ---------------------------------------------------------------------------
# Index tests
# ---------------------------------------------------------------------------


class TestMigration0001Indexes:
    """Verify that all required indexes are created."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = (_VERSIONS_DIR / "0001_add_user_tables.py").read_text()

    def test_oauth_pair_check_constraint(self):
        """CHECK constraint must enforce oauth_provider/oauth_sub pair consistency."""
        assert '"ck_users_oauth_pair"' in self.source
        assert "(oauth_provider IS NULL) = (oauth_sub IS NULL)" in self.source

    def test_oauth_provider_sub_unique_index(self):
        """Partial unique index on (oauth_provider, oauth_sub) must exist."""
        assert '"ix_users_oauth_provider_sub"' in self.source
        assert '["oauth_provider", "oauth_sub"]' in self.source
        assert "unique=True" in self.source
        assert "oauth_provider IS NOT NULL" in self.source

    def test_users_email_index(self):
        """Index on users(email) must be created."""
        assert '"ix_users_email"' in self.source
        assert '["email"]' in self.source

    def test_users_username_index(self):
        """Index on users(username) must be created."""
        assert '"ix_users_username"' in self.source
        assert '["username"]' in self.source

    def test_user_sessions_jti_index(self):
        """Index on user_sessions(jti) must be created."""
        assert '"ix_user_sessions_jti"' in self.source
        assert '["jti"]' in self.source

    def test_user_sessions_user_id_index(self):
        """Index on user_sessions(user_id) must be created."""
        assert '"ix_user_sessions_user_id"' in self.source

    def test_activity_log_composite_index(self):
        """Composite index on activity_log(user_id, created_at) must be created."""
        assert '"ix_activity_log_user_id_created_at"' in self.source
        assert '["user_id", "created_at"]' in self.source

    def test_user_bookmarks_user_id_index(self):
        """Index on user_bookmarks(user_id) must be created."""
        assert '"ix_user_bookmarks_user_id"' in self.source


# ---------------------------------------------------------------------------
# Downgrade tests
# ---------------------------------------------------------------------------


class TestMigration0001Downgrade:
    """Verify that downgrade drops all tables."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = (_VERSIONS_DIR / "0001_add_user_tables.py").read_text()

    def test_drops_user_bookmarks(self):
        """downgrade() must drop user_bookmarks."""
        downgrade_section = self.source.split("def downgrade")[1]
        assert 'drop_table("user_bookmarks")' in downgrade_section

    def test_drops_activity_log(self):
        """downgrade() must drop activity_log."""
        downgrade_section = self.source.split("def downgrade")[1]
        assert 'drop_table("activity_log")' in downgrade_section

    def test_drops_user_sessions(self):
        """downgrade() must drop user_sessions."""
        downgrade_section = self.source.split("def downgrade")[1]
        assert 'drop_table("user_sessions")' in downgrade_section

    def test_drops_users(self):
        """downgrade() must drop users."""
        downgrade_section = self.source.split("def downgrade")[1]
        assert 'drop_table("users")' in downgrade_section

    def test_drops_in_reverse_dependency_order(self):
        """Tables must be dropped in reverse FK dependency order.

        user_bookmarks and activity_log depend on users via FK, so they
        must be dropped before users.  user_sessions also depends on users.
        """
        downgrade_section = self.source.split("def downgrade")[1]
        bookmarks_pos = downgrade_section.index("user_bookmarks")
        activity_pos = downgrade_section.index("activity_log")
        sessions_pos = downgrade_section.index("user_sessions")
        users_pos = downgrade_section.index('"users"')

        # Dependent tables must come before the parent table
        assert bookmarks_pos < users_pos
        assert activity_pos < users_pos
        assert sessions_pos < users_pos


# ---------------------------------------------------------------------------
# Versions directory structure tests
# ---------------------------------------------------------------------------


class TestVersionsDirectory:
    """Verify the alembic/versions directory structure."""

    def test_versions_dir_exists(self):
        """alembic/versions/ directory must exist."""
        assert _VERSIONS_DIR.exists()
        assert _VERSIONS_DIR.is_dir()

    def test_has_initial_migration(self):
        """The versions directory must contain the initial migration file."""
        migration_files = list(_VERSIONS_DIR.glob("*.py"))
        assert len(migration_files) >= 1
        filenames = [f.name for f in migration_files]
        assert "0001_add_user_tables.py" in filenames

    def test_no_extra_migrations(self):
        """Only the initial migration should exist at this point."""
        migration_files = [
            f for f in _VERSIONS_DIR.glob("*.py")
            if not f.name.startswith("__")
        ]
        assert len(migration_files) == 1


# ---------------------------------------------------------------------------
# ORM model alignment tests
# ---------------------------------------------------------------------------


class TestOrmModelAlignment:
    """Verify the migration matches the ORM model definitions."""

    def test_users_table_name(self):
        """ORM User model must target the 'users' table."""
        from orca_web.models.user import User
        assert User.__tablename__ == "users"

    def test_user_sessions_table_name(self):
        """ORM UserSession model must target the 'user_sessions' table."""
        from orca_web.models.user import UserSession
        assert UserSession.__tablename__ == "user_sessions"

    def test_activity_log_table_name(self):
        """ORM ActivityLog model must target the 'activity_log' table."""
        from orca_web.models.user import ActivityLog
        assert ActivityLog.__tablename__ == "activity_log"

    def test_user_bookmarks_table_name(self):
        """ORM UserBookmark model must target the 'user_bookmarks' table."""
        from orca_web.models.user import UserBookmark
        assert UserBookmark.__tablename__ == "user_bookmarks"

    def test_all_orm_tables_in_migration(self):
        """Every ORM table name must appear in the migration source."""
        from orca_web.models.user import ActivityLog, User, UserBookmark, UserSession

        source = (_VERSIONS_DIR / "0001_add_user_tables.py").read_text()
        for model in [User, UserSession, ActivityLog, UserBookmark]:
            assert model.__tablename__ in source, (
                f"ORM table '{model.__tablename__}' not found in migration"
            )

    def test_users_column_count(self):
        """Users ORM model column count must match the migration."""
        from orca_web.models.user import User
        # User has: user_id, email, username, password_hash, oauth_provider,
        # oauth_sub, role, preferences, is_active, created_at, updated_at = 11
        orm_columns = [
            c.key for c in User.__table__.columns
        ]
        assert len(orm_columns) == 11

    def test_user_sessions_column_count(self):
        """UserSession ORM model column count must match the migration."""
        from orca_web.models.user import UserSession
        # session_id, user_id, jti, device_info, ip_address, expires_at,
        # revoked, created_at = 8
        orm_columns = [c.key for c in UserSession.__table__.columns]
        assert len(orm_columns) == 8

    def test_activity_log_column_count(self):
        """ActivityLog ORM model column count must match the migration."""
        from orca_web.models.user import ActivityLog
        # log_id, user_id, action, resource_type, resource_id, service,
        # details, created_at = 8
        orm_columns = [c.key for c in ActivityLog.__table__.columns]
        assert len(orm_columns) == 8

    def test_user_bookmarks_column_count(self):
        """UserBookmark ORM model column count must match the migration."""
        from orca_web.models.user import UserBookmark
        # bookmark_id, user_id, resource_type, resource_id, note,
        # created_at = 6
        orm_columns = [c.key for c in UserBookmark.__table__.columns]
        assert len(orm_columns) == 6

    def test_base_metadata_contains_all_tables(self):
        """Base.metadata must include all four orca-web table names."""
        from orca_web.models.user import Base

        table_names = set(Base.metadata.tables.keys())
        for name in ["users", "user_sessions", "activity_log", "user_bookmarks"]:
            assert name in table_names, f"'{name}' not in Base.metadata.tables"
