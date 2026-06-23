"""Alembic migration environment for orca-web user management tables.

Configures the async SQLAlchemy engine and target metadata for running
migrations against the shared PostgreSQL instance.  Supports both online
(connected to DB via asyncpg) and offline (SQL-generation) modes.

The ``DATABASE_URL`` environment variable, when set, overrides the
``sqlalchemy.url`` value from ``alembic.ini`` so that the same migration
files work in local development, CI, and Docker without editing config.
"""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy.pool import NullPool

from orca_web.models.user import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

if db_url := os.environ.get("DATABASE_URL"):
    config.set_main_option("sqlalchemy.url", db_url)


def run_migrations_offline() -> None:
    """Generate SQL statements without connecting to the database.

    Configures the migration context with the database URL and runs
    all pending migrations in literal-bind mode, emitting DDL as text
    rather than executing it.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Execute migrations using the provided synchronous connection.

    This is the callback passed to ``connection.run_sync()`` inside the
    async migration runner.  It configures the Alembic context and
    runs all pending migrations within a transaction.
    """
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations through it.

    Uses ``NullPool`` so that the connection is closed immediately
    after migrations complete, avoiding leaked connections when running
    as a one-shot script (e.g. ``alembic upgrade head``).
    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in online mode using an async engine.

    Delegates to :func:`run_async_migrations` which creates the async
    engine, executes migrations, and disposes the engine cleanly.
    """
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
