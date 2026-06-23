# Getting Started

> Part of the [Orca](../README.md) meta-learning platform.

---

## Prerequisites

- Docker Engine 24+
- Docker Compose v2
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Node.js 20+ and npm (for orca-ui frontend)

---

## Quick Start with Docker Compose

Starts OrcaMind, OrcaLab, OrcaNet, the Orca Web BFF, PostgreSQL, Redis, MinIO, MLflow, and Prefect in Docker containers.

```bash
git clone https://github.com/AruneemB/orca.git
cd orca

# Install all workspace packages locally
uv venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e packages/orca-shared
uv pip install -e "packages/orcamind[dev]"
uv pip install -e "packages/orcalab[dev]"
uv pip install -e "packages/orcanet[dev]"
uv pip install -e "packages/orca-web[dev]"

# Start backing services (Prefect required by OrcaLab)
docker compose -f docker-compose.dev.yml up -d postgres redis minio mlflow prefect

# Apply database migrations — OrcaMind registry tables (7 tables)
docker compose -f docker-compose.dev.yml run --rm orcamind python scripts/init_db.py

# Apply database migrations — Orca Web user-management tables (4 tables)
export DATABASE_URL="postgresql+asyncpg://orca:orca_dev_secret@localhost:5432/orca_registry"
cd packages/orca-web && alembic upgrade head && cd ../..

# Start OrcaMind, then initialise the Prefect work pool for sweep flows
docker compose -f docker-compose.dev.yml up -d orcamind
python scripts/init_prefect.py

# Start OrcaLab API (waits for OrcaMind to be healthy)
docker compose -f docker-compose.dev.yml up -d orcalab

# Start the Streamlit dashboard (waits for OrcaLab API to be healthy)
docker compose -f docker-compose.dev.yml up -d orcalab-dashboard

# Start OrcaNet (waits for postgres, orcamind, and orcalab to be healthy)
docker compose -f docker-compose.dev.yml up -d orcanet

# Verify
curl http://localhost:8000/health
# → {"status":"healthy","db":true,"faiss":false,"mlflow":true}
curl http://localhost:8001/health
# → {"status":"healthy","db":true,"prefect":true}
curl http://localhost:8002/health
# → {"status":"ok","orcamind":"http://orcamind:8000","orcalab":"http://orcalab:8001"}

# Start the Orca Web BFF (waits for postgres, redis, orcamind, orcalab, orcanet)
docker compose -f docker-compose.dev.yml up -d orca-web
curl http://localhost:8003/health
# → {"status":"healthy","services":{"postgres":true,"redis":true,"orcamind":true,"orcalab":true,"orcanet":true}}
# Dashboard — open in browser: http://localhost:8502
```

Or with Make (starts the full stack including OrcaLab):

```bash
make install
make docker-up
```

> See [Database](DATABASE.md) for migration revision history and adding new revisions.

---

## Local Development Mode

To run OrcaMind outside Docker for hot-reload during development:

```bash
# 1. Start backing services only
docker compose -f docker-compose.dev.yml up -d postgres redis minio mlflow

# 2. Apply migrations (OrcaMind registry tables + Orca Web user tables)
export DATABASE_URL="postgresql+asyncpg://orca:orca_dev_secret@localhost:5432/orca_registry"
cd packages/orcamind && alembic upgrade head && cd ../..
cd packages/orca-web && alembic upgrade head && cd ../..

# 3. Initialise a workspace
orcamind init

# 4. Seed the meta-dataset
python scripts/bootstrap_meta_dataset.py --max-tasks 10 --output-dir data/
```

> See [Database](DATABASE.md#bootstrap-the-meta-dataset) for all seeding CLI flags.

```bash
# 5. Start the OrcaMind API
export DATABASE_URL="postgresql+asyncpg://orca:orca_dev_secret@localhost:5432/orca_registry"
export FAISS_INDEX_PATH="data/orca_task_index.faiss"
orcamind serve --reload
# Interactive docs: http://localhost:8000/docs

# 6. Get model recommendations
orcamind recommend path/to/dataset.csv --top-k 3

# 7. Open the OrcaMind analytics dashboard
orcamind dashboard
# http://localhost:8501

# 8. (Optional) Start the OrcaLab API and dashboard
export PREFECT_API_URL="http://localhost:4200/api"
export ORCAMIND_API_URL="http://localhost:8000"

# Create the Prefect work pool if not already created
python scripts/init_prefect.py

orcalab serve --reload
# Interactive docs: http://localhost:8001/docs

orcalab dashboard
# Streamlit dashboard: http://localhost:8502

# 9. (Optional) Start the OrcaNet transfer agent
export ORCAMIND_API_URL="http://localhost:8000"
export ORCALAB_API_URL="http://localhost:8001"
# export OPENAI_API_KEY="sk-..."  # required for LLM re-ranking

orcanet serve --reload
# Interactive docs: http://localhost:8002/docs

# 10. (Optional) Start the Orca Web BFF
export JWT_SECRET_KEY="dev-secret-change-in-prod"
export ORCAMIND_API_URL="http://localhost:8000"
export ORCALAB_API_URL="http://localhost:8001"
export ORCANET_API_URL="http://localhost:8002"

uvicorn orca_web.api.main:app --host 0.0.0.0 --port 8003 --reload
# Interactive docs: http://localhost:8003/docs

# 11. (Optional) Start the orca-ui frontend
cd packages/orca-ui
npm ci
npm run dev
# React SPA: http://localhost:5173
# Vite proxies /api/* to localhost:8003 (Orca Web BFF)
```
