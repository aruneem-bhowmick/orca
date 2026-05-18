# Getting Started

> Part of the [Orca](../README.md) meta-learning platform.

---

## Prerequisites

- Docker Engine 24+
- Docker Compose v2
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

---

## Quick Start with Docker Compose

The fastest path to a running stack. OrcaMind, OrcaLab, PostgreSQL, Redis, MinIO, MLflow, and Prefect all start together.

```bash
git clone https://github.com/AruneemB/orca.git
cd orca

# Install all workspace packages locally
uv venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e packages/orca-shared
uv pip install -e "packages/orcamind[dev]"
uv pip install -e "packages/orcalab[dev]"
uv pip install -e "packages/orcanet[dev]"

# Start backing services (Prefect required by OrcaLab)
docker compose -f docker-compose.dev.yml up -d postgres redis minio mlflow prefect

# Apply database migrations (creates all 7 registry tables)
docker compose -f docker-compose.dev.yml run --rm orcamind python scripts/init_db.py

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

If you want to run OrcaMind outside Docker (for hot-reload during development):

```bash
# 1. Start backing services only
docker compose -f docker-compose.dev.yml up -d postgres redis minio mlflow

# 2. Apply migrations
export DATABASE_URL="postgresql+asyncpg://orca:orca_dev_secret@localhost:5432/orca_registry"
cd packages/orcamind && alembic upgrade head && cd ../..

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
```
