"""Tests that verify the OrcaNet Docker and deployment configuration is consistent.

These tests are intentionally dependency-free (no Docker, no live services) and
exercise configuration files and environment-variable handling as pure Python.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml  # PyYAML is a transitive dependency via hydra-core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_here = Path(__file__).resolve()
_repo_root = _here.parent
while not (_repo_root / "docker-compose.dev.yml").exists() and _repo_root != _repo_root.parent:
    _repo_root = _repo_root.parent

_COMPOSE_PATH = _repo_root / "docker-compose.dev.yml"
_DOCKERFILE_PATH = _repo_root / "packages" / "orcanet" / "Dockerfile"


def _load_compose() -> dict:
    """Parse docker-compose.dev.yml and return the services dict."""
    assert _COMPOSE_PATH.exists(), f"docker-compose.dev.yml not found at {_COMPOSE_PATH}"
    with _COMPOSE_PATH.open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    return doc.get("services", {})


# ---------------------------------------------------------------------------
# docker-compose.dev.yml tests
# ---------------------------------------------------------------------------


class TestDockerComposeOrcaNet:
    """Verify the ``orcanet`` service block in docker-compose.dev.yml."""

    @pytest.fixture(scope="class")
    def orcanet_svc(self) -> dict:
        """Return the orcanet service block from docker-compose.dev.yml."""
        services = _load_compose()
        assert "orcanet" in services, "'orcanet' service not found in docker-compose.dev.yml"
        return services["orcanet"]

    def test_service_present(self, orcanet_svc: dict) -> None:
        """The orcanet service block exists in the compose file."""
        assert orcanet_svc is not None

    def test_port_mapping(self, orcanet_svc: dict) -> None:
        """OrcaNet is mapped to host port 8002."""
        ports = orcanet_svc.get("ports", [])
        assert any("8002" in str(p) for p in ports), f"Port 8002 not found in: {ports}"

    def test_depends_on_postgres(self, orcanet_svc: dict) -> None:
        """OrcaNet depends on the postgres service."""
        depends = orcanet_svc.get("depends_on", {})
        assert "postgres" in depends

    def test_depends_on_redis(self, orcanet_svc: dict) -> None:
        """OrcaNet depends on the redis service (added by Prompt 14)."""
        depends = orcanet_svc.get("depends_on", {})
        assert "redis" in depends, "redis dependency missing — required by Prompt 14"

    def test_depends_on_orcamind(self, orcanet_svc: dict) -> None:
        """OrcaNet depends on the orcamind service."""
        depends = orcanet_svc.get("depends_on", {})
        assert "orcamind" in depends

    def test_depends_on_orcalab(self, orcanet_svc: dict) -> None:
        """OrcaNet depends on the orcalab service."""
        depends = orcanet_svc.get("depends_on", {})
        assert "orcalab" in depends

    def test_env_var_orcamind_api_url(self, orcanet_svc: dict) -> None:
        """ORCAMIND_API_URL is set in the orcanet service environment."""
        env = orcanet_svc.get("environment", {})
        # env can be a list ("KEY=val") or a dict
        if isinstance(env, list):
            keys = [e.split("=", 1)[0] for e in env]
        else:
            keys = list(env.keys())
        assert "ORCAMIND_API_URL" in keys

    def test_env_var_orcalab_api_url(self, orcanet_svc: dict) -> None:
        """ORCALAB_API_URL is set in the orcanet service environment."""
        env = orcanet_svc.get("environment", {})
        if isinstance(env, list):
            keys = [e.split("=", 1)[0] for e in env]
        else:
            keys = list(env.keys())
        assert "ORCALAB_API_URL" in keys

    def test_env_var_database_url(self, orcanet_svc: dict) -> None:
        """DATABASE_URL is set in the orcanet service environment."""
        env = orcanet_svc.get("environment", {})
        if isinstance(env, list):
            keys = [e.split("=", 1)[0] for e in env]
        else:
            keys = list(env.keys())
        assert "DATABASE_URL" in keys

    def test_healthcheck_present(self, orcanet_svc: dict) -> None:
        """OrcaNet service defines a healthcheck."""
        assert "healthcheck" in orcanet_svc

    def test_healthcheck_port(self, orcanet_svc: dict) -> None:
        """The healthcheck probes port 8002."""
        hc = orcanet_svc["healthcheck"]
        test_cmd = str(hc.get("test", ""))
        assert "8002" in test_cmd, f"Healthcheck does not probe port 8002: {test_cmd}"

    def test_build_context_is_repo_root(self, orcanet_svc: dict) -> None:
        """The build context is the repository root (multi-package build)."""
        build = orcanet_svc.get("build", {})
        context = build.get("context", "") if isinstance(build, dict) else str(build)
        assert context == ".", f"Expected build context '.', got {context!r}"

    def test_dockerfile_path(self, orcanet_svc: dict) -> None:
        """The Dockerfile path points to packages/orcanet/Dockerfile."""
        build = orcanet_svc.get("build", {})
        if isinstance(build, dict):
            dockerfile = build.get("dockerfile", "")
            assert "orcanet" in dockerfile, f"Unexpected Dockerfile: {dockerfile}"


# ---------------------------------------------------------------------------
# Dockerfile tests
# ---------------------------------------------------------------------------


class TestDockerfile:
    """Verify key structural properties of packages/orcanet/Dockerfile."""

    @pytest.fixture(scope="class")
    def dockerfile_text(self) -> str:
        """Return the Dockerfile contents as a string."""
        assert _DOCKERFILE_PATH.exists(), f"Dockerfile not found at {_DOCKERFILE_PATH}"
        return _DOCKERFILE_PATH.read_text(encoding="utf-8")

    def test_multistage_build(self, dockerfile_text: str) -> None:
        """Dockerfile uses a multi-stage build (builder + runtime)."""
        assert "AS builder" in dockerfile_text
        assert "AS runtime" in dockerfile_text

    def test_python_base_image(self, dockerfile_text: str) -> None:
        """Base image is Python 3.11-slim."""
        assert "python:3.11-slim" in dockerfile_text

    def test_uvicorn_cmd(self, dockerfile_text: str) -> None:
        """CMD launches uvicorn on port 8002."""
        assert "uvicorn" in dockerfile_text
        assert "8002" in dockerfile_text

    def test_healthcheck_present(self, dockerfile_text: str) -> None:
        """Dockerfile defines a HEALTHCHECK directive."""
        assert "HEALTHCHECK" in dockerfile_text

    def test_config_copied(self, dockerfile_text: str) -> None:
        """The config/ directory is copied into the runtime image."""
        assert "config" in dockerfile_text


# ---------------------------------------------------------------------------
# Environment variable reading tests
# ---------------------------------------------------------------------------


class TestEnvVarReading:
    """Verify that main.py reads the canonical environment variable names."""

    def test_orcamind_api_url_env_var_name(self) -> None:
        """Lifespan reads ORCAMIND_API_URL (not ORCAMIND_URL)."""
        import inspect

        from orcanet.api import main as main_mod

        source = inspect.getsource(main_mod)
        assert "ORCAMIND_API_URL" in source, (
            "main.py must read ORCAMIND_API_URL to match ecosystem convention"
        )
        assert '"ORCAMIND_URL"' not in source, (
            'main.py must not use the old "ORCAMIND_URL" variable name'
        )

    def test_orcalab_api_url_env_var_name(self) -> None:
        """Lifespan reads ORCALAB_API_URL (not ORCALAB_URL)."""
        import inspect

        from orcanet.api import main as main_mod

        source = inspect.getsource(main_mod)
        assert "ORCALAB_API_URL" in source, (
            "main.py must read ORCALAB_API_URL to match ecosystem convention"
        )
        assert '"ORCALAB_URL"' not in source, (
            'main.py must not use the old "ORCALAB_URL" variable name'
        )

    def test_env_var_overrides_default_orcamind_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ORCAMIND_API_URL env var is read by the lifespan and stored in app.state."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        custom_url = "http://custom-orcamind:9999"
        monkeypatch.setenv("ORCAMIND_API_URL", custom_url)
        # Provide the other required env vars so lifespan can proceed
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://localhost:5432/test")

        from orcanet.api.main import create_app, lifespan

        app = create_app()

        # Make FaissIndex().load() raise FileNotFoundError so the lifespan
        # sets retriever=None and continues rather than crashing.
        faiss_inst = MagicMock()
        faiss_inst.load.side_effect = FileNotFoundError("no index file")

        # Patch all heavyweight startup side-effects so the lifespan can be
        # exercised without a live database, FAISS index, or LLM provider.
        with (
            patch("orcanet.api.main.get_engine", return_value=MagicMock(dispose=AsyncMock())),
            patch("orcanet.api.main.CrossDomainEmbedder", return_value=MagicMock(output_dim=64)),
            patch("orcamind.embedders.similarity.FaissIndex", return_value=faiss_inst),
            patch("orcanet.api.main.FeatureTransfer", return_value=MagicMock()),
            patch("orcanet.api.main.WeightTransfer", return_value=MagicMock()),
            patch("orcanet.api.main.ArchitectureTransfer", return_value=MagicMock()),
            patch("orcanet.api.main.OrcaNetAgent", return_value=MagicMock(llm=MagicMock())),
            patch("orcanet.api.main.QueryExpander"),
            patch("orcanet.api.main.LLMRanker"),
            patch("orca_shared.clients.orcamind_client.OrcaMindClient", return_value=MagicMock(aclose=AsyncMock())),
            patch("orca_shared.clients.orcalab_client.OrcaLabClient", return_value=MagicMock(aclose=AsyncMock())),
        ):
            async def run():
                async with lifespan(app):
                    return app.state.orcamind_url

            result = asyncio.run(run())

        assert result == custom_url

    def test_env_var_overrides_default_orcalab_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ORCALAB_API_URL env var is read by the lifespan and stored in app.state."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        custom_url = "http://custom-orcalab:9998"
        monkeypatch.setenv("ORCALAB_API_URL", custom_url)
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://localhost:5432/test")

        from orcanet.api.main import create_app, lifespan

        app = create_app()

        faiss_inst = MagicMock()
        faiss_inst.load.side_effect = FileNotFoundError("no index file")

        with (
            patch("orcanet.api.main.get_engine", return_value=MagicMock(dispose=AsyncMock())),
            patch("orcanet.api.main.CrossDomainEmbedder", return_value=MagicMock(output_dim=64)),
            patch("orcamind.embedders.similarity.FaissIndex", return_value=faiss_inst),
            patch("orcanet.api.main.FeatureTransfer", return_value=MagicMock()),
            patch("orcanet.api.main.WeightTransfer", return_value=MagicMock()),
            patch("orcanet.api.main.ArchitectureTransfer", return_value=MagicMock()),
            patch("orcanet.api.main.OrcaNetAgent", return_value=MagicMock(llm=MagicMock())),
            patch("orcanet.api.main.QueryExpander"),
            patch("orcanet.api.main.LLMRanker"),
            patch("orca_shared.clients.orcamind_client.OrcaMindClient", return_value=MagicMock(aclose=AsyncMock())),
            patch("orca_shared.clients.orcalab_client.OrcaLabClient", return_value=MagicMock(aclose=AsyncMock())),
        ):
            async def run():
                async with lifespan(app):
                    return app.state.orcalab_url

            result = asyncio.run(run())

        assert result == custom_url
