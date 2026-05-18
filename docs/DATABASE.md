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

---

## Schema Reference

All tables use PostgreSQL UUID primary keys (`gen_random_uuid()` default via SQLAlchemy). Timestamps are `timestamptz` with a server-side `NOW()` default.

ORM source: `packages/orca-shared/orca_shared/registry/models.py`

---

### `embeddings`

Stores task embedding vectors produced by the statistical or neural embedder.

| Column             | Type              | Nullable | Notes                                                  |
|--------------------|-------------------|----------|--------------------------------------------------------|
| `embedding_id`     | `UUID` PK         | no       |                                                        |
| `task_id`          | `UUID` FK → tasks | yes      | Back-reference to the task this embedding belongs to   |
| `embedding_type`   | `VARCHAR(50)`     | yes      | `"statistical"`, `"neural"`, etc.                      |
| `embedding_vector` | `FLOAT[]`         | yes      | Raw ARRAY of floats; length == `dimension`             |
| `dimension`        | `INTEGER`         | yes      | Vector dimensionality (25 for statistical, 64 for neural) |
| `model_version`    | `VARCHAR(50)`     | yes      | Embedder version tag                                   |
| `created_at`       | `TIMESTAMPTZ`     | no       | Server-default `NOW()`                                 |

---

### `tasks`

Registry of ML datasets/tasks. The `tasks ↔ embeddings` relationship is bidirectional: a task may have many embeddings (history) and one active `embedding_id`.

| Column         | Type                   | Nullable | Notes                                                                             |
|----------------|------------------------|----------|-----------------------------------------------------------------------------------|
| `task_id`      | `UUID` PK              | no       |                                                                                   |
| `name`         | `VARCHAR(255)`         | no       |                                                                                   |
| `domain`       | `VARCHAR(100)`         | yes      | e.g. `"classification"`, `"regression"`                                           |
| `task_type`    | `VARCHAR(50)`          | no       | e.g. `"multiclass"`, `"binary"`, `"regression"`                                   |
| `n_samples`    | `INTEGER`              | yes      |                                                                                   |
| `n_features`   | `INTEGER`              | yes      |                                                                                   |
| `n_classes`    | `INTEGER`              | yes      | `NULL` for regression tasks                                                       |
| `dataset_uri`  | `TEXT`                 | yes      | S3/MinIO path or local file URI                                                   |
| `metadata`     | `JSONB`                | yes      | Column alias for ORM attribute `task_metadata` (avoids DeclarativeBase collision) |
| `embedding_id` | `UUID` FK → embeddings | yes      | Currently active embedding; **deferred FK** `fk_tasks_embedding_id` — see note below |
| `created_at`   | `TIMESTAMPTZ`          | no       | Server-default `NOW()`                                                            |
| `updated_at`   | `TIMESTAMPTZ`          | no       | Updated on every row write via `onupdate=NOW()`                                   |

**Deferred FK note**: `fk_tasks_embedding_id` is defined with `use_alter=True` (Alembic emits `ADD CONSTRAINT … DEFERRABLE INITIALLY DEFERRED`). This resolves the circular dependency between `tasks` and `embeddings` — both can be inserted in the same transaction without ordering constraints.

---

### `models`

Registered model architectures and their hyperparameter configurations.

| Column            | Type           | Nullable | Notes                                          |
|-------------------|----------------|----------|------------------------------------------------|
| `model_id`        | `UUID` PK      | no       |                                                |
| `name`            | `VARCHAR(255)` | no       |                                                |
| `architecture`    | `VARCHAR(100)` | yes      | e.g. `"resnet18"`, `"random_forest"`           |
| `config`          | `JSONB`        | no       | Full hyperparameter configuration dict         |
| `parameter_count` | `BIGINT`       | yes      | Total trainable parameters                     |
| `flops`           | `BIGINT`       | yes      | Estimated FLOPs for one forward pass           |
| `created_at`      | `TIMESTAMPTZ`  | no       | Server-default `NOW()`                         |

---

### `experiments`

A single training run (trial), linking a task to a model configuration with lifecycle status and accumulated metrics.

| Column            | Type                    | Nullable | Notes                                                                                |
|-------------------|-------------------------|----------|--------------------------------------------------------------------------------------|
| `experiment_id`   | `UUID` PK               | no       |                                                                                      |
| `task_id`         | `UUID` FK → tasks       | yes      |                                                                                      |
| `model_id`        | `UUID` FK → models      | yes      |                                                                                      |
| `training_config` | `JSONB`                 | yes      | `TrainingConfig` dict: `batch_size`, `lr`, `epochs`, `optimizer`, `scheduler`, etc.  |
| `status`          | `VARCHAR(50)`           | yes      | `pending`, `queued`, `running`, `completed`, `failed`, `cancelled`                   |
| `mlflow_run_id`   | `VARCHAR(255)`          | yes      | MLflow run ID for metric/artifact lookup                                             |
| `started_at`      | `TIMESTAMPTZ`           | yes      |                                                                                      |
| `completed_at`    | `TIMESTAMPTZ`           | yes      |                                                                                      |
| `created_by`      | `VARCHAR(100)`          | yes      | Free-form creator tag                                                                |
| `metrics`         | `JSONB`                 | yes      | **Added in revision 0002.** Accumulates per-epoch snapshots: `{"loss": float, "epoch": int}`. Written atomically by `ExperimentRepository.update_metrics()` with a row-level lock (`.with_for_update()`) to prevent concurrent epoch writes from clobbering each other. `NULL` before the first epoch completes; treated as `{}` by the WebSocket handler. |

---

### `performances`

Granular per-metric, per-epoch records for each experiment. Complements `experiments.metrics` (which stores the most recent epoch snapshot) with full history.

| Column         | Type                       | Nullable | Notes                                             |
|----------------|----------------------------|----------|---------------------------------------------------|
| `performance_id` | `UUID` PK                | no       |                                                   |
| `experiment_id`  | `UUID` FK → experiments  | yes      |                                                   |
| `metric_name`    | `VARCHAR(100)`           | yes      | e.g. `"accuracy"`, `"loss"`, `"f1"`              |
| `metric_value`   | `FLOAT`                  | yes      |                                                   |
| `epoch`          | `INTEGER`                | yes      | Training step / epoch number                      |
| `is_final`       | `BOOLEAN`                | no       | `true` for the final evaluation metric; default `false` |
| `metadata`       | `JSONB`                  | yes      | Column alias for ORM attribute `perf_metadata`    |
| `recorded_at`    | `TIMESTAMPTZ`            | no       | Server-default `NOW()`                            |

---

### `transfer_mappings`

Pairwise source→target task transfer scores, used by OrcaMind's warm-start transfer selector.

| Column           | Type              | Nullable | Notes                                           |
|------------------|-------------------|----------|-------------------------------------------------|
| `mapping_id`     | `UUID` PK         | no       |                                                 |
| `source_task_id` | `UUID` FK → tasks | yes      |                                                 |
| `target_task_id` | `UUID` FK → tasks | yes      |                                                 |
| `transfer_score` | `FLOAT`           | yes      | Higher = more transferable                      |
| `transfer_type`  | `VARCHAR(50)`     | yes      | e.g. `"warm_start"`, `"maml"`, `"reptile"`      |
| `metadata`       | `JSONB`           | yes      | Column alias for ORM attribute `mapping_metadata` |
| `created_at`     | `TIMESTAMPTZ`     | no       | Server-default `NOW()`                          |

---

### `search_spaces`

Hyperparameter search space definitions, optionally organized as a tree via self-referential `parent_id`.

| Column            | Type                    | Nullable | Notes                                               |
|-------------------|-------------------------|----------|-----------------------------------------------------|
| `search_space_id` | `UUID` PK               | no       |                                                     |
| `name`            | `VARCHAR(255)`          | yes      |                                                     |
| `definition`      | `JSONB`                 | no       | Full space definition: `name`, `description`, `parameters[]` |
| `parent_id`       | `UUID` FK → self        | yes      | Self-referential FK for nested / composed spaces    |
| `created_at`      | `TIMESTAMPTZ`           | no       | Server-default `NOW()`                              |
