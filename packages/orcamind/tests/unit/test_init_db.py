"""Unit tests for scripts/init_db.py."""
from __future__ import annotations

import os
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_init_db_calls_upgrade_head(init_db_module: ModuleType) -> None:
    mock_cmd = MagicMock()
    mock_cfg = MagicMock()
    with patch.object(init_db_module, "command", mock_cmd), \
         patch.object(init_db_module, "Config", return_value=mock_cfg), \
         patch.dict(os.environ, {}, clear=True):
        init_db_module.main()

    mock_cmd.upgrade.assert_called_once_with(mock_cfg, "head")


def test_init_db_locates_alembic_ini_relative_to_script(init_db_module: ModuleType) -> None:
    captured: list[str] = []

    def _capture(path: str) -> MagicMock:
        captured.append(path)
        return MagicMock()

    with patch.object(init_db_module, "Config", side_effect=_capture), \
         patch.object(init_db_module, "command"), \
         patch.dict(os.environ, {}, clear=True):
        init_db_module.main()

    assert captured, "Config() was never called"
    assert captured[0].endswith("alembic.ini"), f"Unexpected ini path: {captured[0]}"


def test_init_db_uses_database_url_env(init_db_module: ModuleType) -> None:
    test_url = "postgresql+asyncpg://u:p@host:5432/db"
    set_calls: list[tuple] = []

    def _fake_config(path: str) -> MagicMock:
        cfg = MagicMock()
        cfg.set_main_option.side_effect = lambda k, v: set_calls.append((k, v))
        return cfg

    with patch.object(init_db_module, "Config", side_effect=_fake_config), \
         patch.object(init_db_module, "command"), \
         patch.dict(os.environ, {"DATABASE_URL": test_url}, clear=True):
        init_db_module.main()

    assert ("sqlalchemy.url", test_url) in set_calls


def test_init_db_does_not_set_url_when_env_absent(init_db_module: ModuleType) -> None:
    set_calls: list[tuple] = []

    def _fake_config(path: str) -> MagicMock:
        cfg = MagicMock()
        cfg.set_main_option.side_effect = lambda k, v: set_calls.append((k, v))
        return cfg

    with patch.object(init_db_module, "Config", side_effect=_fake_config), \
         patch.object(init_db_module, "command"), \
         patch.dict(os.environ, {}, clear=True):
        init_db_module.main()

    assert not any(k == "sqlalchemy.url" for k, _ in set_calls)


def test_init_db_exits_nonzero_on_failure(init_db_module: ModuleType) -> None:
    mock_cmd = MagicMock()
    mock_cmd.upgrade.side_effect = RuntimeError("connection refused")

    with patch.object(init_db_module, "command", mock_cmd), \
         patch.object(init_db_module, "Config", return_value=MagicMock()), \
         patch.dict(os.environ, {}, clear=True):
        with pytest.raises(SystemExit) as exc_info:
            init_db_module.main()

    assert exc_info.value.code != 0
