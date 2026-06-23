"""Create user management tables for the orca-web BFF.

Creates the four tables owned by orca-web: users, user_sessions,
activity_log, and user_bookmarks.  These live in the same PostgreSQL
instance used by the rest of the platform but are managed exclusively
by the BFF gateway for authentication, session tracking, activity
logging, and user bookmarks.

Revision ID: 0001
Revises:
Create Date: 2026-06-22 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create user management tables with indexes."""
    # --- users ---
    op.create_table(
        "users",
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("username", sa.String(100), unique=True, nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=True),
        sa.Column("oauth_provider", sa.String(50), nullable=True),
        sa.Column("oauth_sub", sa.String(255), nullable=True),
        sa.Column(
            "role", sa.String(20), nullable=False, server_default="user",
        ),
        sa.Column("preferences", postgresql.JSONB(), nullable=True),
        sa.Column(
            "is_active", sa.Boolean(), nullable=False, server_default="true",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_users_email", "users", ["email"])
    op.create_index("ix_users_username", "users", ["username"])

    # --- user_sessions ---
    op.create_table(
        "user_sessions",
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.user_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("jti", sa.String(255), unique=True, nullable=False),
        sa.Column("device_info", sa.Text(), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column(
            "expires_at", sa.DateTime(timezone=True), nullable=False,
        ),
        sa.Column(
            "revoked", sa.Boolean(), nullable=False, server_default="false",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_user_sessions_jti", "user_sessions", ["jti"])
    op.create_index("ix_user_sessions_user_id", "user_sessions", ["user_id"])

    # --- activity_log ---
    op.create_table(
        "activity_log",
        sa.Column(
            "log_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.user_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("resource_type", sa.String(50), nullable=True),
        sa.Column("resource_id", sa.String(255), nullable=True),
        sa.Column("service", sa.String(50), nullable=True),
        sa.Column("details", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_activity_log_user_id_created_at",
        "activity_log",
        ["user_id", "created_at"],
    )

    # --- user_bookmarks ---
    op.create_table(
        "user_bookmarks",
        sa.Column(
            "bookmark_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.user_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("resource_type", sa.String(50), nullable=False),
        sa.Column("resource_id", sa.String(255), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_user_bookmarks_user_id", "user_bookmarks", ["user_id"])


def downgrade() -> None:
    """Drop all user management tables in reverse dependency order."""
    op.drop_table("user_bookmarks")
    op.drop_table("activity_log")
    op.drop_table("user_sessions")
    op.drop_table("users")
