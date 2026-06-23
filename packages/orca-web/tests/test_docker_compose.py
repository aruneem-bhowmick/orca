"""Tests for the orca-web service definition in docker-compose.dev.yml.

These tests verify that the Docker Compose configuration correctly wires the
BFF service with proper ports, environment variables, dependencies, and health
checks — all without requiring Docker to be installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture(scope="module")
def compose(repo_root: Path) -> dict[str, Any]:
    """Parse docker-compose.dev.yml and return the full document."""
    text = (repo_root / "docker-compose.dev.yml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


@pytest.fixture(scope="module")
def services(compose: dict[str, Any]) -> dict[str, Any]:
    """Return the services dict from the compose document."""
    return compose["services"]


@pytest.fixture(scope="module")
def orca_web_svc(services: dict[str, Any]) -> dict[str, Any]:
    """Return the orca-web service block."""
    return services["orca-web"]


def _env_keys(svc: dict[str, Any]) -> list[str]:
    """Extract environment variable names from a service block.

    Handles both dict-style (``KEY: value``) and list-style
    (``KEY=value``) environment declarations.
    """
    env = svc.get("environment", {})
    if isinstance(env, dict):
        return list(env.keys())
    return [e.split("=", 1)[0] for e in env]


# ---------------------------------------------------------------------------
# Service presence
# ---------------------------------------------------------------------------


class TestServicePresence:
    """Verify the orca-web service exists in the compose file."""

    def test_service_exists(self, services: dict[str, Any]) -> None:
        """The orca-web service block must be present."""
        assert "orca-web" in services


# ---------------------------------------------------------------------------
# Build configuration
# ---------------------------------------------------------------------------


class TestBuildConfiguration:
    """Verify the build context and Dockerfile path."""

    def test_build_context_is_repo_root(self, orca_web_svc: dict[str, Any]) -> None:
        """The build context must be the repository root for multi-package builds."""
        build = orca_web_svc.get("build", {})
        context = build.get("context", "") if isinstance(build, dict) else str(build)
        assert context == ".", f"Expected build context '.', got {context!r}"

    def test_dockerfile_path(self, orca_web_svc: dict[str, Any]) -> None:
        """The Dockerfile path must point to packages/orca-web/Dockerfile."""
        build = orca_web_svc.get("build", {})
        assert isinstance(build, dict), "build must be a mapping"
        dockerfile = build.get("dockerfile", "")
        assert dockerfile == "packages/orca-web/Dockerfile", (
            f"Expected 'packages/orca-web/Dockerfile', got {dockerfile!r}"
        )


# ---------------------------------------------------------------------------
# Port mapping
# ---------------------------------------------------------------------------


class TestPortMapping:
    """Verify the service is exposed on the correct port."""

    def test_publishes_port_8003(self, orca_web_svc: dict[str, Any]) -> None:
        """The BFF must be mapped to host port 8003."""
        ports = orca_web_svc.get("ports", [])
        assert "8003:8003" in ports, f"Expected '8003:8003' in ports, got: {ports}"


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------


class TestEnvironmentVariables:
    """Verify all required environment variables are set."""

    def test_has_database_url(self, orca_web_svc: dict[str, Any]) -> None:
        """DATABASE_URL must be set for PostgreSQL connectivity."""
        assert "DATABASE_URL" in _env_keys(orca_web_svc)

    def test_has_redis_url(self, orca_web_svc: dict[str, Any]) -> None:
        """REDIS_URL must be set for session and cache operations."""
        assert "REDIS_URL" in _env_keys(orca_web_svc)

    def test_has_orcamind_api_url(self, orca_web_svc: dict[str, Any]) -> None:
        """ORCAMIND_API_URL must be set for upstream proxy routing."""
        assert "ORCAMIND_API_URL" in _env_keys(orca_web_svc)

    def test_has_orcalab_api_url(self, orca_web_svc: dict[str, Any]) -> None:
        """ORCALAB_API_URL must be set for upstream proxy routing."""
        assert "ORCALAB_API_URL" in _env_keys(orca_web_svc)

    def test_has_orcanet_api_url(self, orca_web_svc: dict[str, Any]) -> None:
        """ORCANET_API_URL must be set for upstream proxy routing."""
        assert "ORCANET_API_URL" in _env_keys(orca_web_svc)

    def test_has_jwt_secret_key(self, orca_web_svc: dict[str, Any]) -> None:
        """JWT_SECRET_KEY must be set for token signing."""
        assert "JWT_SECRET_KEY" in _env_keys(orca_web_svc)

    def test_has_frontend_url(self, orca_web_svc: dict[str, Any]) -> None:
        """FRONTEND_URL must be set for OAuth redirect handling."""
        assert "FRONTEND_URL" in _env_keys(orca_web_svc)

    def test_has_cors_origins(self, orca_web_svc: dict[str, Any]) -> None:
        """CORS_ORIGINS must be set for cross-origin browser requests."""
        assert "CORS_ORIGINS" in _env_keys(orca_web_svc)

    def test_database_url_uses_asyncpg(self, orca_web_svc: dict[str, Any]) -> None:
        """DATABASE_URL must use the asyncpg driver for async SQLAlchemy."""
        env = orca_web_svc.get("environment", {})
        db_url = env.get("DATABASE_URL", "") if isinstance(env, dict) else ""
        assert "asyncpg" in db_url

    def test_upstream_urls_use_docker_hostnames(self, orca_web_svc: dict[str, Any]) -> None:
        """Upstream service URLs must use Docker service hostnames, not localhost."""
        env = orca_web_svc.get("environment", {})
        if isinstance(env, dict):
            assert "orcamind" in env.get("ORCAMIND_API_URL", "")
            assert "orcalab" in env.get("ORCALAB_API_URL", "")
            assert "orcanet" in env.get("ORCANET_API_URL", "")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


class TestDependencies:
    """Verify service dependency declarations with health conditions."""

    def test_depends_on_postgres(self, orca_web_svc: dict[str, Any]) -> None:
        """The BFF must depend on postgres for its user-management tables."""
        deps = orca_web_svc.get("depends_on", {})
        assert "postgres" in deps

    def test_depends_on_redis(self, orca_web_svc: dict[str, Any]) -> None:
        """The BFF must depend on redis for session storage and caching."""
        deps = orca_web_svc.get("depends_on", {})
        assert "redis" in deps

    def test_depends_on_orcamind(self, orca_web_svc: dict[str, Any]) -> None:
        """The BFF must depend on orcamind for proxy routing."""
        deps = orca_web_svc.get("depends_on", {})
        assert "orcamind" in deps

    def test_depends_on_orcalab(self, orca_web_svc: dict[str, Any]) -> None:
        """The BFF must depend on orcalab for proxy routing and WebSocket relay."""
        deps = orca_web_svc.get("depends_on", {})
        assert "orcalab" in deps

    def test_depends_on_orcanet(self, orca_web_svc: dict[str, Any]) -> None:
        """The BFF must depend on orcanet for proxy routing."""
        deps = orca_web_svc.get("depends_on", {})
        assert "orcanet" in deps

    def test_postgres_uses_service_healthy(self, orca_web_svc: dict[str, Any]) -> None:
        """The postgres dependency must use condition: service_healthy."""
        deps = orca_web_svc.get("depends_on", {})
        pg = deps.get("postgres", {})
        assert pg.get("condition") == "service_healthy"

    def test_redis_uses_service_healthy(self, orca_web_svc: dict[str, Any]) -> None:
        """The redis dependency must use condition: service_healthy."""
        deps = orca_web_svc.get("depends_on", {})
        rd = deps.get("redis", {})
        assert rd.get("condition") == "service_healthy"

    def test_orcamind_uses_service_healthy(self, orca_web_svc: dict[str, Any]) -> None:
        """The orcamind dependency must use condition: service_healthy."""
        deps = orca_web_svc.get("depends_on", {})
        mind = deps.get("orcamind", {})
        assert mind.get("condition") == "service_healthy"

    def test_orcalab_uses_service_healthy(self, orca_web_svc: dict[str, Any]) -> None:
        """The orcalab dependency must use condition: service_healthy."""
        deps = orca_web_svc.get("depends_on", {})
        lab = deps.get("orcalab", {})
        assert lab.get("condition") == "service_healthy"

    def test_orcanet_uses_service_healthy(self, orca_web_svc: dict[str, Any]) -> None:
        """The orcanet dependency must use condition: service_healthy."""
        deps = orca_web_svc.get("depends_on", {})
        net = deps.get("orcanet", {})
        assert net.get("condition") == "service_healthy"


# ---------------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------------


class TestHealthcheck:
    """Verify the service healthcheck configuration."""

    def test_has_healthcheck(self, orca_web_svc: dict[str, Any]) -> None:
        """The service must define a healthcheck block."""
        assert "healthcheck" in orca_web_svc

    def test_healthcheck_targets_port_8003(self, orca_web_svc: dict[str, Any]) -> None:
        """The healthcheck must probe port 8003."""
        hc = orca_web_svc["healthcheck"]
        test_cmd = str(hc.get("test", ""))
        assert "8003" in test_cmd, f"Healthcheck does not target port 8003: {test_cmd}"

    def test_healthcheck_has_start_period(self, orca_web_svc: dict[str, Any]) -> None:
        """The healthcheck must define a start_period to allow for startup time."""
        hc = orca_web_svc["healthcheck"]
        assert "start_period" in hc
