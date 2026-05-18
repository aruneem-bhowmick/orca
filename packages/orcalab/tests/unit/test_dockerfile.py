"""Unit tests for packages/orcalab/Dockerfile structure."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def dockerfile_text(repo_root: Path) -> str:
    return (repo_root / "packages" / "orcalab" / "Dockerfile").read_text()


class TestMultiStageBuild:
    def test_builder_stage_declared(self, dockerfile_text: str) -> None:
        assert "AS builder" in dockerfile_text

    def test_runtime_stage_declared(self, dockerfile_text: str) -> None:
        assert "AS runtime" in dockerfile_text


class TestBuilderStage:
    def test_installs_uv(self, dockerfile_text: str) -> None:
        assert "uv" in dockerfile_text

    def test_copies_pyproject_toml(self, dockerfile_text: str) -> None:
        assert "pyproject.toml" in dockerfile_text


class TestRuntimeStage:
    def test_copies_venv_from_builder(self, dockerfile_text: str) -> None:
        assert "--from=builder" in dockerfile_text
        assert "/opt/venv" in dockerfile_text

    def test_copies_orcalab_source_from_builder(self, dockerfile_text: str) -> None:
        assert "packages/orcalab/orcalab" in dockerfile_text

    def test_pythonunbuffered_set(self, dockerfile_text: str) -> None:
        assert "PYTHONUNBUFFERED=1" in dockerfile_text

    def test_healthcheck_present(self, dockerfile_text: str) -> None:
        assert "HEALTHCHECK" in dockerfile_text

    def test_expose_8001(self, dockerfile_text: str) -> None:
        assert "EXPOSE 8001" in dockerfile_text

    def test_cmd_runs_uvicorn(self, dockerfile_text: str) -> None:
        assert "uvicorn" in dockerfile_text

    def test_cmd_references_app_entrypoint(self, dockerfile_text: str) -> None:
        assert "orcalab.api.main:app" in dockerfile_text

    def test_cmd_binds_correct_port(self, dockerfile_text: str) -> None:
        assert "8001" in dockerfile_text
