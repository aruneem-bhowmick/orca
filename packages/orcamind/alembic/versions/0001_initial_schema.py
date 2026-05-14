"""Initial schema — all seven registry tables.

Revision ID: 0001
Revises:
Create Date: 2026-05-14 00:00:00.000000

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
    # search_spaces — self-referential, no external deps
    op.create_table(
        "search_spaces",
        sa.Column("search_space_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("definition", postgresql.JSONB(), nullable=False),
        sa.Column(
            "parent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("search_spaces.search_space_id"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # models — no external deps
    op.create_table(
        "models",
        sa.Column("model_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("architecture", sa.String(100), nullable=True),
        sa.Column("config", postgresql.JSONB(), nullable=False),
        sa.Column("parameter_count", sa.BigInteger(), nullable=True),
        sa.Column("flops", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # tasks — created without embedding_id FK to break circular dep with embeddings
    op.create_table(
        "tasks",
        sa.Column("task_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("domain", sa.String(100), nullable=True),
        sa.Column("task_type", sa.String(50), nullable=False),
        sa.Column("n_samples", sa.Integer(), nullable=True),
        sa.Column("n_features", sa.Integer(), nullable=True),
        sa.Column("n_classes", sa.Integer(), nullable=True),
        sa.Column("dataset_uri", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column("embedding_id", postgresql.UUID(as_uuid=True), nullable=True),
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

    # embeddings — depends on tasks
    op.create_table(
        "embeddings",
        sa.Column("embedding_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "task_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tasks.task_id"),
            nullable=True,
        ),
        sa.Column("embedding_type", sa.String(50), nullable=True),
        sa.Column("embedding_vector", postgresql.ARRAY(sa.Float()), nullable=True),
        sa.Column("dimension", sa.Integer(), nullable=True),
        sa.Column("model_version", sa.String(50), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Now add the deferred FK from tasks.embedding_id → embeddings.embedding_id
    op.create_foreign_key(
        "fk_tasks_embedding_id",
        "tasks",
        "embeddings",
        ["embedding_id"],
        ["embedding_id"],
    )

    # experiments — depends on tasks + models
    op.create_table(
        "experiments",
        sa.Column("experiment_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "task_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tasks.task_id"),
            nullable=True,
        ),
        sa.Column(
            "model_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("models.model_id"),
            nullable=True,
        ),
        sa.Column("training_config", postgresql.JSONB(), nullable=True),
        sa.Column("status", sa.String(50), nullable=True),
        sa.Column("mlflow_run_id", sa.String(255), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(100), nullable=True),
    )

    # performances — depends on experiments
    op.create_table(
        "performances",
        sa.Column("performance_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "experiment_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("experiments.experiment_id"),
            nullable=True,
        ),
        sa.Column("metric_name", sa.String(100), nullable=True),
        sa.Column("metric_value", sa.Float(), nullable=True),
        sa.Column("epoch", sa.Integer(), nullable=True),
        sa.Column("is_final", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # transfer_mappings — depends on tasks (twice)
    op.create_table(
        "transfer_mappings",
        sa.Column("mapping_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "source_task_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tasks.task_id"),
            nullable=True,
        ),
        sa.Column(
            "target_task_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tasks.task_id"),
            nullable=True,
        ),
        sa.Column("transfer_score", sa.Float(), nullable=True),
        sa.Column("transfer_type", sa.String(50), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("transfer_mappings")
    op.drop_table("performances")
    op.drop_table("experiments")
    op.drop_constraint("fk_tasks_embedding_id", "tasks", type_="foreignkey")
    op.drop_table("embeddings")
    op.drop_table("tasks")
    op.drop_table("models")
    op.drop_table("search_spaces")
