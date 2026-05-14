"""Unit tests for alembic configuration and migration file structure."""
from __future__ import annotations

import configparser
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def alembic_root(orcamind_pkg_dir: Path) -> Path:
    return orcamind_pkg_dir


@pytest.fixture(scope="module")
def migration_file(alembic_root: Path) -> Path:
    return alembic_root / "alembic" / "versions" / "0001_initial_schema.py"


@pytest.fixture(scope="module")
def migration_text(migration_file: Path) -> str:
    return migration_file.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def env_text(alembic_root: Path) -> str:
    return (alembic_root / "alembic" / "env.py").read_text(encoding="utf-8")


# ── alembic.ini ───────────────────────────────────────────────────────────────

def test_alembic_ini_exists(alembic_root: Path) -> None:
    assert (alembic_root / "alembic.ini").is_file()


def test_alembic_ini_has_correct_script_location(alembic_root: Path) -> None:
    cfg = configparser.ConfigParser()
    cfg.read(str(alembic_root / "alembic.ini"))
    assert cfg.get("alembic", "script_location") == "alembic"


def test_alembic_ini_has_sqlalchemy_url(alembic_root: Path) -> None:
    cfg = configparser.ConfigParser()
    cfg.read(str(alembic_root / "alembic.ini"))
    url = cfg.get("alembic", "sqlalchemy.url")
    assert url.startswith("postgresql+asyncpg://")


# ── alembic/env.py ────────────────────────────────────────────────────────────

def test_alembic_env_py_exists(alembic_root: Path) -> None:
    assert (alembic_root / "alembic" / "env.py").is_file()


def test_env_py_uses_async_engine(env_text: str) -> None:
    assert "async_engine_from_config" in env_text


def test_env_py_imports_base(env_text: str) -> None:
    assert "from orca_shared.registry.models import Base" in env_text


def test_env_py_sets_target_metadata(env_text: str) -> None:
    assert "target_metadata = Base.metadata" in env_text


def test_env_py_reads_database_url_env(env_text: str) -> None:
    assert "DATABASE_URL" in env_text


def test_env_py_uses_null_pool(env_text: str) -> None:
    assert "NullPool" in env_text


# ── alembic/versions/ ────────────────────────────────────────────────────────

def test_alembic_versions_dir_exists(alembic_root: Path) -> None:
    assert (alembic_root / "alembic" / "versions").is_dir()


def test_initial_migration_exists(migration_file: Path) -> None:
    assert migration_file.is_file()


def test_migration_has_upgrade_function(migration_text: str) -> None:
    assert "def upgrade()" in migration_text


def test_migration_has_downgrade_function(migration_text: str) -> None:
    assert "def downgrade()" in migration_text


def test_migration_revision_is_0001(migration_text: str) -> None:
    assert 'revision: str = "0001"' in migration_text


def test_migration_down_revision_is_none(migration_text: str) -> None:
    assert "down_revision" in migration_text
    assert "None" in migration_text


# ── Migration covers all seven tables ────────────────────────────────────────

EXPECTED_TABLES = [
    "tasks",
    "models",
    "experiments",
    "performances",
    "embeddings",
    "transfer_mappings",
    "search_spaces",
]


@pytest.mark.parametrize("table_name", EXPECTED_TABLES)
def test_migration_creates_table(table_name: str, migration_text: str) -> None:
    assert f'"{table_name}"' in migration_text, (
        f"Table '{table_name}' not found in migration"
    )


def test_migration_handles_circular_fk(migration_text: str) -> None:
    assert "fk_tasks_embedding_id" in migration_text
    assert "create_foreign_key" in migration_text


def test_migration_drops_all_tables_in_downgrade(migration_text: str) -> None:
    for table in EXPECTED_TABLES:
        assert f'drop_table("{table}")' in migration_text or \
               f"drop_table('{table}')" in migration_text or \
               f'"{table}"' in migration_text.split("def downgrade")[1]
