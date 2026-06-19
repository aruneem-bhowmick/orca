"""Tests for scripts/init_prefect.py module import and __main__ guard."""

from __future__ import annotations

from types import ModuleType
from unittest.mock import patch

import pytest


class TestModuleImport:
    """Verify the module loads and exposes the expected public function."""

    def test_create_orcalab_pool_exists(self, init_prefect: ModuleType) -> None:
        assert hasattr(init_prefect, "create_orcalab_pool")

    def test_create_orcalab_pool_is_callable(
        self, init_prefect: ModuleType
    ) -> None:
        assert callable(init_prefect.create_orcalab_pool)


class TestCreateOrcalabPool:
    """Cover the subprocess call shape (mirrors the orcalab suite) and the
    CalledProcessError propagation that the orcalab suite does not test."""

    def test_check_true_propagates_subprocess_failure(
        self, init_prefect: ModuleType
    ) -> None:
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "prefect")):
            with pytest.raises(subprocess.CalledProcessError):
                init_prefect.create_orcalab_pool()

    def test_returns_none_on_success(self, init_prefect: ModuleType) -> None:
        with patch("subprocess.run") as mock_run:
            result = init_prefect.create_orcalab_pool()
        assert result is None
        mock_run.assert_called_once()
