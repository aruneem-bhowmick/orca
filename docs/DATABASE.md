# Database

> Part of the [Orca](../README.md) meta-learning platform.

---

## Migrations

OrcaMind uses [Alembic](https://alembic.sqlalchemy.org/) to manage the PostgreSQL schema. The migration environment is configured for SQLAlchemy's async engine (`asyncpg` driver) using `NullPool` so connections close after migrations complete.

### Apply migrations

```bash
# Inside Docker (preferred)
docker compose -f docker-compose.dev.yml run --rm orcamind python scripts/init_db.py

# Or directly with Alembic (local dev, DATABASE_URL must be set)
export DATABASE_URL="postgresql+asyncpg://orca:orca_dev_secret@localhost:5432/orca_registry"
cd packages/orcamind
alembic upgrade head
```

`scripts/init_db.py` resolves `alembic.ini` relative to its own path, reads `DATABASE_URL` from the environment, and exits non-zero on any failure — making it safe to call as a Docker Compose pre-start step.

### Revision history


| Revision | Description                                                                                                                                                                |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0001`   | Initial schema — all 7 registry tables in FK-safe creation order; deferred `fk_tasks_embedding_id` to resolve the `tasks ↔ embeddings` circular dependency |
| `0002`   | Add nullable JSONB `metrics` column to `experiments` — stores per-epoch snapshots (`{"loss": float, "epoch": int}`) written by `ExperimentRepository.update_metrics()`; nullable so existing rows default to `NULL` (treated as `{}` by the repository and WebSocket handler); fully reversible via `downgrade` |


To generate a new revision after ORM changes:

```bash
cd packages/orcamind
alembic revision --autogenerate -m "describe your change"
alembic upgrade head
```

---

## Bootstrap the Meta-Dataset

Seeds the registry with real benchmark tasks from OpenML:

```bash
export DATABASE_URL="postgresql+asyncpg://orca:orca_dev_secret@localhost:5432/orca_registry"

python scripts/bootstrap_meta_dataset.py \
  --suites cc18 ctr23 \
  --max-tasks 20 \
  --output-dir data/
```


| Flag                | Default             | Description                             |
| ------------------- | ------------------- | --------------------------------------- |
| `--suites`          | `cc18 ctr23`        | OpenML benchmark suites to download     |
| `--max-tasks INT`   | all                 | Cap on tasks per suite                  |
| `--output-dir PATH` | `data/`             | Directory for FAISS index output        |
| `--db-url URL`      | from `DATABASE_URL` | Override database connection            |
| `--dry-run`         | off                 | Parse + embed without writing to the DB |


**What it does:**

1. Downloads **OpenML CC-18** (classification, ≤72 tasks) and/or **CTR-23** (regression)
2. For each task: extracts features from the raw dataset and computes a 25-dim statistical embedding
3. Runs 5 baseline models (Logistic Regression, Random Forest, XGBoost, SVM, KNN) with 5-fold cross-validation, skipping SVM/SVR for datasets >10,000 samples
4. Persists `Task`, `Model`, `Experiment`, `Performance` rows to PostgreSQL via the repository layer
5. Adds each task embedding to an in-memory FAISS cosine-similarity index
6. Saves the completed index to `{output-dir}/orca_task_index.faiss`

After seeding, `GET /api/v1/tasks` and the Recommendation Explorer will return real data.
