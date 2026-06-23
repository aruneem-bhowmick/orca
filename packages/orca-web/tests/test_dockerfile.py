"""Tests for packages/orca-web/Dockerfile structure.

These tests verify key structural properties of the Dockerfile without
requiring Docker to be installed or any containers to be running.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def dockerfile_text(repo_root: Path) -> str:
    """Return the Dockerfile contents as a string."""
    path = repo_root / "packages" / "orca-web" / "Dockerfile"
    assert path.exists(), f"Dockerfile not found at {path}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Multi-stage build structure
# ---------------------------------------------------------------------------


class TestMultiStageBuild:
    """Verify that the Dockerfile uses a two-stage build pattern."""

    def test_builder_stage_declared(self, dockerfile_text: str) -> None:
        """The builder stage must be declared with 'AS builder'."""
        assert "AS builder" in dockerfile_text

    def test_runtime_stage_declared(self, dockerfile_text: str) -> None:
        """The runtime stage must be declared with 'AS runtime'."""
        assert "AS runtime" in dockerfile_text


# ---------------------------------------------------------------------------
# Builder stage
# ---------------------------------------------------------------------------


class TestBuilderStage:
    """Verify that the builder stage installs dependencies correctly."""

    def test_python_base_image(self, dockerfile_text: str) -> None:
        """The base image should be Python 3.11-slim."""
        assert "python:3.11-slim" in dockerfile_text

    def test_installs_uv(self, dockerfile_text: str) -> None:
        """The builder must install the uv package manager."""
        assert "uv" in dockerfile_text

    def test_copies_orca_shared_manifest(self, dockerfile_text: str) -> None:
        """The builder must copy the orca-shared pyproject.toml for layer caching."""
        assert "orca-shared/pyproject.toml" in dockerfile_text

    def test_copies_orca_web_manifest(self, dockerfile_text: str) -> None:
        """The builder must copy the orca-web pyproject.toml for layer caching."""
        assert "orca-web/pyproject.toml" in dockerfile_text

    def test_creates_venv(self, dockerfile_text: str) -> None:
        """The builder must create a virtual environment at /opt/venv."""
        assert "uv venv /opt/venv" in dockerfile_text

    def test_copies_orca_shared_source(self, dockerfile_text: str) -> None:
        """The builder must copy the orca-shared source package."""
        assert "packages/orca-shared/" in dockerfile_text

    def test_copies_orca_web_source(self, dockerfile_text: str) -> None:
        """The builder must copy the orca-web source package."""
        assert "packages/orca-web/" in dockerfile_text


# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------


class TestRuntimeStage:
    """Verify runtime stage configuration for production readiness."""

    def test_copies_venv_from_builder(self, dockerfile_text: str) -> None:
        """The runtime must copy the venv from the builder stage."""
        assert "--from=builder" in dockerfile_text
        assert "/opt/venv" in dockerfile_text

    def test_creates_non_root_user(self, dockerfile_text: str) -> None:
        """The runtime must create and switch to a non-root orca user."""
        assert "orca" in dockerfile_text
        assert "USER orca" in dockerfile_text

    def test_pythonunbuffered_set(self, dockerfile_text: str) -> None:
        """PYTHONUNBUFFERED must be set for proper log output."""
        assert "PYTHONUNBUFFERED=1" in dockerfile_text

    def test_pythondontwritebytecode_set(self, dockerfile_text: str) -> None:
        """PYTHONDONTWRITEBYTECODE must be set to avoid .pyc in the image."""
        assert "PYTHONDONTWRITEBYTECODE=1" in dockerfile_text

    def test_copies_alembic_ini(self, dockerfile_text: str) -> None:
        """The runtime must copy alembic.ini for migration support."""
        assert "alembic.ini" in dockerfile_text

    def test_copies_alembic_directory(self, dockerfile_text: str) -> None:
        """The runtime must copy the alembic/ directory for migration scripts."""
        assert "alembic" in dockerfile_text

    def test_expose_8003(self, dockerfile_text: str) -> None:
        """The Dockerfile must expose port 8003."""
        assert "EXPOSE 8003" in dockerfile_text

    def test_healthcheck_present(self, dockerfile_text: str) -> None:
        """The Dockerfile must define a HEALTHCHECK directive."""
        assert "HEALTHCHECK" in dockerfile_text

    def test_healthcheck_port(self, dockerfile_text: str) -> None:
        """The healthcheck must probe port 8003."""
        # Find the HEALTHCHECK line and verify it targets port 8003
        lines = dockerfile_text.splitlines()
        hc_lines = [ln for ln in lines if "HEALTHCHECK" in ln or "httpx" in ln]
        hc_text = " ".join(hc_lines)
        assert "8003" in hc_text

    def test_cmd_runs_uvicorn(self, dockerfile_text: str) -> None:
        """The CMD must launch uvicorn."""
        assert "uvicorn" in dockerfile_text

    def test_cmd_references_app_entrypoint(self, dockerfile_text: str) -> None:
        """The CMD must reference the orca_web.api.main:app entrypoint."""
        assert "orca_web.api.main:app" in dockerfile_text

    def test_cmd_binds_correct_port(self, dockerfile_text: str) -> None:
        """The CMD must bind to port 8003."""
        # Look for the port in the CMD line
        lines = dockerfile_text.splitlines()
        cmd_lines = [ln for ln in lines if "CMD" in ln and "uvicorn" in ln]
        assert any("8003" in ln for ln in cmd_lines)

    def test_workdir_is_app(self, dockerfile_text: str) -> None:
        """The runtime WORKDIR must be /app."""
        assert "WORKDIR /app" in dockerfile_text
