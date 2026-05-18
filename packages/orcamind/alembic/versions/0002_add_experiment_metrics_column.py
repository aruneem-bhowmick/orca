"""Add nullable JSONB metrics column to experiments table.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-18 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "experiments",
        sa.Column("metrics", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("experiments", "metrics")
