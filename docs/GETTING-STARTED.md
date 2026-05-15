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

The fastest path to a running stack. OrcaMind, PostgreSQL, Redis, MinIO, and MLflow all start together.

```bash
git clone https://github.com/AruneemB/orca.git
cd orca

# Install all workspace packages locally
uv venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e packages/orca-shared
uv pip install -e "packages/orcamind[dev]"

# Start backing services
docker compose -f docker-compose.dev.yml up -d postgres redis minio mlflow

# Apply database migrations (creates all 7 registry tables)
docker compose -f docker-compose.dev.yml run --rm orcamind python scripts/init_db.py

# Start OrcaMind
docker compose -f docker-compose.dev.yml up -d orcamind

# Verify
curl http://localhost:8000/health
# → {"status":"healthy","db":true,"faiss":false,"mlflow":true}
```

Or with Make:

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
# 5. Start the API
export DATABASE_URL="postgresql+asyncpg://orca:orca_dev_secret@localhost:5432/orca_registry"
export FAISS_INDEX_PATH="data/orca_task_index.faiss"
orcamind serve --reload
# Interactive docs: http://localhost:8000/docs

# 6. Get model recommendations
orcamind recommend path/to/dataset.csv --top-k 3

# 7. Open the analytics dashboard
orcamind dashboard
# http://localhost:8501
```
