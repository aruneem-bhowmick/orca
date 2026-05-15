"""Integration tests for docker-compose development services.

These tests verify live service connectivity and are skipped automatically
when Docker is unavailable or the stack is not running.

Run manually after `make docker-up`:
    pytest packages/orcamind/tests/integration/ -v
"""

from __future__ import annotations

import socket

import pytest


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


requires_postgres = pytest.mark.skipif(
    not _port_open("localhost", 5432),
    reason="PostgreSQL not reachable on localhost:5432 — run `make docker-up` first",
)

requires_redis = pytest.mark.skipif(
    not _port_open("localhost", 6379),
    reason="Redis not reachable on localhost:6379 — run `make docker-up` first",
)

requires_minio = pytest.mark.skipif(
    not _port_open("localhost", 9000),
    reason="MinIO not reachable on localhost:9000 — run `make docker-up` first",
)

requires_mlflow = pytest.mark.skipif(
    not _port_open("localhost", 5000),
    reason="MLflow not reachable on localhost:5000 — run `make docker-up` first",
)

requires_prefect = pytest.mark.skipif(
    not _port_open("localhost", 4200),
    reason="Prefect not reachable on localhost:4200 — run `make docker-up` first",
)

requires_orcamind = pytest.mark.skipif(
    not _port_open("localhost", 8000),
    reason="OrcaMind not reachable on localhost:8000 — run `make docker-up` first",
)


# ── PostgreSQL ────────────────────────────────────────────────────────────────

@requires_postgres
def test_postgres_tcp_reachable() -> None:
    assert _port_open("localhost", 5432)


@requires_postgres
def test_postgres_accepts_connection() -> None:
    import asyncpg
    import asyncio

    async def _connect():
        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            user="orca",
            password="orca_dev_secret",
            database="orca_registry",
        )
        result = await conn.fetchval("SELECT 1")
        await conn.close()
        return result

    assert asyncio.run(_connect()) == 1


# ── Redis ─────────────────────────────────────────────────────────────────────

@requires_redis
def test_redis_tcp_reachable() -> None:
    assert _port_open("localhost", 6379)


@requires_redis
def test_redis_ping() -> None:
    import redis

    client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    assert client.ping() is True
    client.close()


# ── MinIO ─────────────────────────────────────────────────────────────────────

@requires_minio
def test_minio_api_reachable() -> None:
    assert _port_open("localhost", 9000)


@requires_minio
def test_minio_console_reachable() -> None:
    assert _port_open("localhost", 9001)


@requires_minio
def test_minio_health_endpoint() -> None:
    import httpx

    resp = httpx.get("http://localhost:9000/minio/health/live", timeout=5)
    assert resp.status_code == 200


# ── MLflow ────────────────────────────────────────────────────────────────────

@requires_mlflow
def test_mlflow_api_reachable() -> None:
    assert _port_open("localhost", 5000)


@requires_mlflow
def test_mlflow_health_endpoint() -> None:
    import httpx

    resp = httpx.get("http://localhost:5000/health", timeout=10)
    assert resp.status_code == 200


@requires_mlflow
def test_mlflow_can_list_experiments() -> None:
    import mlflow

    mlflow.set_tracking_uri("http://localhost:5000")
    experiments = mlflow.search_experiments()
    assert isinstance(experiments, list)


# ── Prefect ───────────────────────────────────────────────────────────────────

@requires_prefect
def test_prefect_api_reachable() -> None:
    assert _port_open("localhost", 4200)


@requires_prefect
def test_prefect_health_endpoint() -> None:
    import httpx

    resp = httpx.get("http://localhost:4200/api/health", timeout=10)
    assert resp.status_code == 200


# ── OrcaMind ──────────────────────────────────────────────────────────────────

@requires_orcamind
def test_orcamind_tcp_reachable() -> None:
    assert _port_open("localhost", 8000)


@requires_orcamind
def test_orcamind_root_endpoint_returns_200() -> None:
    import httpx

    resp = httpx.get("http://localhost:8000/", timeout=10)
    assert resp.status_code == 200


@requires_orcamind
def test_orcamind_root_endpoint_body() -> None:
    import httpx

    body = httpx.get("http://localhost:8000/", timeout=10).json()
    assert body.get("name") == "OrcaMind"
    assert body.get("status") == "ok"


@requires_orcamind
def test_orcamind_health_endpoint() -> None:
    import httpx

    resp = httpx.get("http://localhost:8000/health", timeout=10)
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "healthy"


@requires_orcamind
def test_orcamind_health_has_required_keys() -> None:
    import httpx

    body = httpx.get("http://localhost:8000/health", timeout=10).json()
    assert "status" in body
    assert "db" in body
    assert "faiss" in body
    assert "mlflow" in body


@requires_orcamind
def test_orcamind_health_status_is_valid() -> None:
    import httpx

    # A fresh stack with no FAISS index loaded reports "degraded"; only
    # "healthy" after bootstrap_meta_dataset.py has been run.
    body = httpx.get("http://localhost:8000/health", timeout=10).json()
    assert body.get("status") in ("healthy", "degraded")


@requires_orcamind
def test_orcamind_health_db_is_true() -> None:
    import httpx

    body = httpx.get("http://localhost:8000/health", timeout=10).json()
    assert body.get("db") is True


@requires_orcamind
def test_orcamind_tasks_endpoint_returns_list() -> None:
    import httpx

    resp = httpx.get("http://localhost:8000/api/v1/tasks", timeout=10)
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    assert "total" in body
    assert isinstance(body["items"], list)


@requires_orcamind
def test_orcamind_recommend_endpoint_accepts_post() -> None:
    import httpx

    payload = {
        "n_samples": 100,
        "n_features": 10,
        "task_type": "classification",
    }
    resp = httpx.post(
        "http://localhost:8000/api/v1/recommend-model",
        json=payload,
        timeout=10,
    )
    # 200 (recommendation returned) or 422 (validation) are both acceptable;
    # a 5xx would indicate a server-side crash.
    assert resp.status_code < 500
