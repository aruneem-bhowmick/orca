# Deployment

> Part of the [Orca](../README.md) meta-learning platform.

This guide covers deploying the full Orca stack using Docker Compose. For local development setup, see [GETTING-STARTED.md](GETTING-STARTED.md).

---

## Service Topology

```text
PostgreSQL 15  ─────────────────────────────────────┐
Redis 7         ─────────────────────────────────┐   │
MinIO           ────────────────────────────┐    │   │
                                             ↓    ↓   ↓
MLflow  ─────────────────────────────────────────────┤
Prefect ──────────────────────────────────────────── ┤
                                             ↓    ↓   ↓
OrcaMind (port 8000) ────────────────────────────────┤
                                             ↓    ↓   ↓
OrcaLab  (port 8001) ────────────────────────────────┤
OrcaLab Dashboard (port 8502) ←─ depends on OrcaLab
```

---

## Environment Variables

### Infrastructure services

| Variable                 | Used by          | Default (dev)               | Description                              |
|--------------------------|------------------|-----------------------------|------------------------------------------|
| `POSTGRES_USER`          | postgres         | `orca`                      | Database superuser                       |
| `POSTGRES_PASSWORD`      | postgres         | `orca_dev_secret`           | Database password                        |
| `POSTGRES_DB`            | postgres         | `orca_registry`             | Database name                            |
| `MINIO_ROOT_USER`        | minio            | `orca_minio_user`           | MinIO root user (S3 API)                 |
| `MINIO_ROOT_PASSWORD`    | minio            | `orca_minio_secret`         | MinIO root password                      |
| `MLFLOW_S3_ENDPOINT_URL` | mlflow           | `http://minio:9000`         | Points MLflow artifact storage at MinIO  |
| `AWS_ACCESS_KEY_ID`      | mlflow           | `orca_minio_user`           | MinIO access key used by MLflow          |
| `AWS_SECRET_ACCESS_KEY`  | mlflow           | `orca_minio_secret`         | MinIO secret key used by MLflow          |

### OrcaMind (port 8000)

| Variable              | Required | Default (dev)                                          | Description                                   |
|-----------------------|----------|--------------------------------------------------------|-----------------------------------------------|
| `DATABASE_URL`        | yes      | `postgresql+asyncpg://orca:orca_dev_secret@postgres:5432/orca_registry` | Async PostgreSQL connection string |
| `REDIS_URL`           | yes      | `redis://redis:6379`                                   | Redis connection string                       |
| `MINIO_ENDPOINT`      | yes      | `minio:9000`                                           | MinIO host:port (no scheme)                   |
| `MINIO_ACCESS_KEY`    | yes      | `orca_minio_user`                                      | MinIO access key                              |
| `MINIO_SECRET_KEY`    | yes      | `orca_minio_secret`                                    | MinIO secret key                              |
| `MLFLOW_TRACKING_URI` | yes      | `http://mlflow:5000`                                   | MLflow tracking server URL                    |
| `ORCA_DATA_DIR`       | no       | `/data`                                                | Host path for local artifact storage          |
| `CORS_ORIGINS`        | no       | —                                                      | Comma-separated allowed CORS origins; deny-all when unset |

### OrcaLab (port 8001)

| Variable              | Required | Default (dev)                                          | Description                                   |
|-----------------------|----------|--------------------------------------------------------|-----------------------------------------------|
| `DATABASE_URL`        | yes      | `postgresql+asyncpg://orca:orca_dev_secret@postgres:5432/orca_registry` | Async PostgreSQL connection string |
| `MLFLOW_TRACKING_URI` | yes      | `http://mlflow:5000`                                   | MLflow tracking server URL                    |
| `PREFECT_API_URL`     | no       | —                                                      | Prefect server API URL; sweep Prefect triggers disabled when unset |
| `ORCAMIND_API_URL`    | no       | —                                                      | OrcaMind service base URL; meta-informed search degrades gracefully when unset |
| `CORS_ORIGINS`        | no       | —                                                      | Comma-separated allowed CORS origins; deny-all when unset |

### OrcaLab Dashboard (port 8502)

| Variable          | Required | Default (dev)              | Description                           |
|-------------------|----------|----------------------------|---------------------------------------|
| `ORCALAB_API_URL` | no       | `http://localhost:8001`    | OrcaLab API base URL for the Streamlit dashboard |

---

## Service Startup Order

Services must start in dependency order. Docker Compose `depends_on` with `condition: service_healthy` enforces this automatically:

```
1. postgres        ← healthcheck: pg_isready
2. redis           ← healthcheck: redis-cli ping
3. minio           ← healthcheck: curl /minio/health/live
4. mlflow          ← healthcheck: curl /health     (depends on postgres, minio)
5. prefect         ← healthcheck: curl /api/health  (depends on postgres)
6. orcamind        ← healthcheck: httpx GET /health  (depends on postgres, redis, minio, mlflow)
7. orcalab         ← healthcheck: httpx GET /health  (depends on all above + orcamind)
8. orcalab-dashboard   ← no healthcheck (depends on orcalab)
```

Before OrcaMind starts, run the Alembic migrations:

```bash
docker compose -f docker-compose.dev.yml run --rm orcamind python scripts/init_db.py
```

---

## Starting the Stack

```bash
# Apply database migrations first (required before OrcaMind starts)
docker compose -f docker-compose.dev.yml run --rm orcamind python scripts/init_db.py

# Start all services
docker compose -f docker-compose.dev.yml up -d

# Initialize the Prefect work pool (first run only)
docker compose -f docker-compose.dev.yml run --rm orcalab python scripts/init_prefect.py

# Tail logs
docker compose -f docker-compose.dev.yml logs -f orcamind orcalab
```

> After schema changes on a running stack, re-run `init_db.py` (`docker compose ... run --rm orcamind python scripts/init_db.py`) while services are up — the migration runner uses `NullPool` and exits cleanly without disrupting live connections.

The `init_prefect.py` script creates the `orcalab-pool` Prefect work pool (type: `process`) that the `meta_informed_sweep` deployment requires. This is a one-time operation per Prefect server instance.

---

## Health Checks

Once all services are up, verify health:

```bash
# OrcaMind
curl http://localhost:8000/health
# {"status": "ok", "faiss": false}  ← faiss: false until bootstrap_meta_dataset runs

# OrcaLab
curl http://localhost:8001/health
# {"status": "ok", "prefect": "http://prefect:4200/api"}

# MLflow
curl http://localhost:5000/health

# Prefect
curl http://localhost:4200/api/health
```

**OrcaMind FAISS index**: `faiss: false` means the `/recommend-model` and `/similar-tasks` endpoints return 503. Run the bootstrap script to seed the registry and build the index:

```bash
docker compose -f docker-compose.dev.yml run --rm orcamind \
  python scripts/bootstrap_meta_dataset.py --suites cc18 --max-tasks 20 --output-dir /data
```

---

## Monitoring

| Interface             | URL                         | Notes                                       |
|-----------------------|-----------------------------|---------------------------------------------|
| OrcaMind API docs     | http://localhost:8000/docs  | Swagger UI for all OrcaMind endpoints       |
| OrcaLab API docs      | http://localhost:8001/docs  | Swagger UI for all OrcaLab endpoints        |
| OrcaLab Dashboard     | http://localhost:8502       | Streamlit UI — Live Experiments, Search Progress, Results Explorer, Meta-Analysis |
| MLflow UI             | http://localhost:5000       | Experiment runs, metrics, model registry    |
| Prefect UI            | http://localhost:4200       | Flow runs, work pools, deployments          |
| MinIO Console         | http://localhost:9001       | Object storage browser                      |

---

## Prefect Worker Setup

OrcaLab orchestrates hyperparameter sweeps as Prefect flows. After `init_prefect.py` creates the `orcalab-pool` work pool, start a worker to process flow runs:

```bash
prefect worker start --pool orcalab-pool
```

For Docker-based workers, the worker process must have network access to the OrcaLab API and the Prefect server. Scale horizontally by starting multiple worker processes against the same pool.

---

## Production Considerations

### Secrets management

Replace the dev-mode inline passwords with proper secrets before deploying to any shared environment:

- Use Docker secrets (`secrets:` block) or environment files (`.env`) excluded from version control.
- Rotate `POSTGRES_PASSWORD`, `MINIO_ROOT_PASSWORD`, and `MINIO_SECRET_KEY`.
- Set `CORS_ORIGINS` to your frontend origin(s); if unset, CORS requests are denied by default.

### Persistence

The Compose file declares named volumes for PostgreSQL, Redis, MinIO, and the OrcaMind data directory. For production:

- Mount PostgreSQL data to a reliable network volume or managed database service.
- Use external MinIO storage or replace MinIO with S3/GCS and point `MINIO_ENDPOINT` + credentials accordingly.
- Prefect uses PostgreSQL for its own metadata (`PREFECT_API_DATABASE_CONNECTION_URL`); this can share the same cluster with a separate database.

### Scaling OrcaLab

OrcaLab's FastAPI service is stateless except for `app.state.sweeps` (an in-memory dict keyed by `sweep_id`). Horizontal scaling behind a load balancer requires either:
- Replacing the in-memory sweep store with a Redis-backed store, or
- Using sticky sessions so sweep status reads hit the same instance that created the sweep.

OrcaMind is stateless (all state in PostgreSQL + FAISS on disk) and can be scaled horizontally freely.

### Log retention

The Compose file sets JSON-file logging with `max-size: 10m` and `max-file: 3` per service. Adjust these limits or switch to a centralized logging driver (`fluentd`, `loki`) for production deployments.
