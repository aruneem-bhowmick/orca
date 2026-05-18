"""Unit tests for docker-compose.dev.yml service configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture(scope="module")
def compose(repo_root: Path) -> dict[str, Any]:
    text = (repo_root / "docker-compose.dev.yml").read_text()
    return yaml.safe_load(text)


@pytest.fixture(scope="module")
def services(compose: dict[str, Any]) -> dict[str, Any]:
    return compose["services"]


@pytest.fixture(scope="module")
def orcalab_svc(services: dict[str, Any]) -> dict[str, Any]:
    return services["orcalab"]


@pytest.fixture(scope="module")
def dashboard_svc(services: dict[str, Any]) -> dict[str, Any]:
    return services["orcalab-dashboard"]


class TestOrcalabService:
    def test_service_exists(self, services: dict[str, Any]) -> None:
        assert "orcalab" in services

    def test_publishes_port_8001(self, orcalab_svc: dict[str, Any]) -> None:
        ports = orcalab_svc.get("ports", [])
        assert any("8001" in str(p) for p in ports)

    def test_has_database_url(self, orcalab_svc: dict[str, Any]) -> None:
        env = orcalab_svc.get("environment", {})
        env_keys = list(env.keys()) if isinstance(env, dict) else [e.split("=")[0] for e in env]
        assert "DATABASE_URL" in env_keys

    def test_has_mlflow_tracking_uri(self, orcalab_svc: dict[str, Any]) -> None:
        env = orcalab_svc.get("environment", {})
        env_keys = list(env.keys()) if isinstance(env, dict) else [e.split("=")[0] for e in env]
        assert "MLFLOW_TRACKING_URI" in env_keys

    def test_has_prefect_api_url(self, orcalab_svc: dict[str, Any]) -> None:
        env = orcalab_svc.get("environment", {})
        env_keys = list(env.keys()) if isinstance(env, dict) else [e.split("=")[0] for e in env]
        assert "PREFECT_API_URL" in env_keys

    def test_has_orcamind_api_url(self, orcalab_svc: dict[str, Any]) -> None:
        env = orcalab_svc.get("environment", {})
        env_keys = list(env.keys()) if isinstance(env, dict) else [e.split("=")[0] for e in env]
        assert "ORCAMIND_API_URL" in env_keys

    def test_depends_on_postgres(self, orcalab_svc: dict[str, Any]) -> None:
        deps = orcalab_svc.get("depends_on", {})
        assert "postgres" in deps

    def test_depends_on_redis(self, orcalab_svc: dict[str, Any]) -> None:
        deps = orcalab_svc.get("depends_on", {})
        assert "redis" in deps

    def test_depends_on_mlflow(self, orcalab_svc: dict[str, Any]) -> None:
        deps = orcalab_svc.get("depends_on", {})
        assert "mlflow" in deps

    def test_depends_on_prefect(self, orcalab_svc: dict[str, Any]) -> None:
        deps = orcalab_svc.get("depends_on", {})
        assert "prefect" in deps

    def test_depends_on_orcamind(self, orcalab_svc: dict[str, Any]) -> None:
        deps = orcalab_svc.get("depends_on", {})
        assert "orcamind" in deps

    def test_has_healthcheck(self, orcalab_svc: dict[str, Any]) -> None:
        assert "healthcheck" in orcalab_svc


class TestOrcalabDashboardService:
    def test_service_exists(self, services: dict[str, Any]) -> None:
        assert "orcalab-dashboard" in services

    def test_publishes_port_8502(self, dashboard_svc: dict[str, Any]) -> None:
        ports = dashboard_svc.get("ports", [])
        assert any("8502" in str(p) for p in ports)

    def test_depends_on_orcalab(self, dashboard_svc: dict[str, Any]) -> None:
        deps = dashboard_svc.get("depends_on", {})
        assert "orcalab" in deps

    def test_has_orcalab_api_url(self, dashboard_svc: dict[str, Any]) -> None:
        env = dashboard_svc.get("environment", {})
        env_keys = list(env.keys()) if isinstance(env, dict) else [e.split("=")[0] for e in env]
        assert "ORCALAB_API_URL" in env_keys

    def test_command_runs_streamlit(self, dashboard_svc: dict[str, Any]) -> None:
        cmd = dashboard_svc.get("command", [])
        cmd_str = " ".join(str(c) for c in cmd)
        assert "streamlit" in cmd_str

    def test_command_targets_visualization_app(self, dashboard_svc: dict[str, Any]) -> None:
        cmd = dashboard_svc.get("command", [])
        cmd_str = " ".join(str(c) for c in cmd)
        assert "visualization" in cmd_str

    def test_command_binds_port_8502(self, dashboard_svc: dict[str, Any]) -> None:
        cmd = dashboard_svc.get("command", [])
        cmd_str = " ".join(str(c) for c in cmd)
        assert "8502" in cmd_str
