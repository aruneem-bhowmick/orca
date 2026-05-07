"""Validate docker-compose.dev.yml structure, services, images, and ports."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="module")
def compose(repo_root: Path) -> dict:
    return yaml.safe_load((repo_root / "docker-compose.dev.yml").read_text())


@pytest.fixture(scope="module")
def services(compose: dict) -> dict:
    return compose["services"]


# ── File exists ───────────────────────────────────────────────────────────────

def test_docker_compose_file_exists(repo_root: Path) -> None:
    assert (repo_root / "docker-compose.dev.yml").is_file()


def test_compose_is_valid_yaml(compose: dict) -> None:
    assert isinstance(compose, dict)


# ── All required services are defined ─────────────────────────────────────────

REQUIRED_SERVICES = ["postgres", "redis", "minio", "mlflow", "prefect"]


@pytest.mark.parametrize("service", REQUIRED_SERVICES)
def test_service_exists(service: str, services: dict) -> None:
    assert service in services, f"Service '{service}' not found in docker-compose.dev.yml"


# ── Postgres ──────────────────────────────────────────────────────────────────

def test_postgres_image(services: dict) -> None:
    assert services["postgres"]["image"] == "postgres:15-alpine"


def test_postgres_port(services: dict) -> None:
    assert "5432:5432" in services["postgres"]["ports"]


def test_postgres_db_name(services: dict) -> None:
    assert services["postgres"]["environment"]["POSTGRES_DB"] == "orca_registry"


def test_postgres_user(services: dict) -> None:
    assert services["postgres"]["environment"]["POSTGRES_USER"] == "orca"


def test_postgres_has_healthcheck(services: dict) -> None:
    assert "healthcheck" in services["postgres"]


def test_postgres_healthcheck_uses_pg_isready(services: dict) -> None:
    test_cmd = services["postgres"]["healthcheck"]["test"]
    assert any("pg_isready" in str(c) for c in test_cmd)


def test_postgres_has_named_volume(services: dict) -> None:
    volumes = services["postgres"]["volumes"]
    assert any("postgres-data" in v for v in volumes)


# ── Redis ─────────────────────────────────────────────────────────────────────

def test_redis_image(services: dict) -> None:
    assert services["redis"]["image"] == "redis:7-alpine"


def test_redis_port(services: dict) -> None:
    assert "6379:6379" in services["redis"]["ports"]


def test_redis_has_healthcheck(services: dict) -> None:
    assert "healthcheck" in services["redis"]


def test_redis_healthcheck_uses_ping(services: dict) -> None:
    test_cmd = services["redis"]["healthcheck"]["test"]
    assert any("ping" in str(c) for c in test_cmd)


# ── MinIO ─────────────────────────────────────────────────────────────────────

def test_minio_image(services: dict) -> None:
    assert "minio" in services["minio"]["image"]


def test_minio_api_port(services: dict) -> None:
    assert "9000:9000" in services["minio"]["ports"]


def test_minio_console_port(services: dict) -> None:
    assert "9001:9001" in services["minio"]["ports"]


def test_minio_has_healthcheck(services: dict) -> None:
    assert "healthcheck" in services["minio"]


def test_minio_has_named_volume(services: dict) -> None:
    volumes = services["minio"]["volumes"]
    assert any("minio-data" in v for v in volumes)


# ── MLflow ────────────────────────────────────────────────────────────────────

def test_mlflow_image_is_pinned(services: dict) -> None:
    assert services["mlflow"]["image"] == "ghcr.io/mlflow/mlflow:v2.10.0"


def test_mlflow_port(services: dict) -> None:
    assert "5000:5000" in services["mlflow"]["ports"]


def test_mlflow_minio_endpoint_configured(services: dict) -> None:
    env = services["mlflow"]["environment"]
    assert "MLFLOW_S3_ENDPOINT_URL" in env
    assert "minio:9000" in env["MLFLOW_S3_ENDPOINT_URL"]


def test_mlflow_depends_on_postgres(services: dict) -> None:
    assert "postgres" in services["mlflow"]["depends_on"]


def test_mlflow_depends_on_minio(services: dict) -> None:
    assert "minio" in services["mlflow"]["depends_on"]


def test_mlflow_has_healthcheck(services: dict) -> None:
    assert "healthcheck" in services["mlflow"]


# ── Prefect ───────────────────────────────────────────────────────────────────

def test_prefect_image(services: dict) -> None:
    assert "prefect" in services["prefect"]["image"].lower()
    assert "2" in services["prefect"]["image"]


def test_prefect_port(services: dict) -> None:
    assert "4200:4200" in services["prefect"]["ports"]


def test_prefect_depends_on_postgres(services: dict) -> None:
    assert "postgres" in services["prefect"]["depends_on"]


def test_prefect_has_healthcheck(services: dict) -> None:
    assert "healthcheck" in services["prefect"]


# ── Volumes and networking ────────────────────────────────────────────────────

def test_named_volumes_defined(compose: dict) -> None:
    volumes = compose.get("volumes", {})
    for v in ("postgres-data", "redis-data", "minio-data"):
        assert v in volumes, f"Named volume '{v}' not declared in top-level volumes"


def test_network_name(compose: dict) -> None:
    networks = compose.get("networks", {})
    default_net = networks.get("default", {})
    assert default_net.get("name") == "orca-dev-network"


# ── Application service stubs are commented out ───────────────────────────────

def test_orcamind_service_is_not_active(services: dict) -> None:
    assert "orcamind" not in services, (
        "orcamind service should remain commented-out until its image is buildable"
    )


def test_orcalab_service_is_not_active(services: dict) -> None:
    assert "orcalab" not in services


def test_orcanet_service_is_not_active(services: dict) -> None:
    assert "orcanet" not in services
