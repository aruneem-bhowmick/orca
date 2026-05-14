# Orca

**A unified meta-learning platform. Teach machines how to learn, not just what to learn.**

---

Orca is a monorepo meta-learning ecosystem built around one idea: **prior experiments are a dataset, and we should learn from them**. Rather than starting every new ML task from scratch, Orca accumulates knowledge across tasks, embeds what it has seen, and uses that memory to recommend models, warm-start training, and guide hyperparameter search.

The ecosystem is composed of three interconnected services — **OrcaMind**, **OrcaLab**, and **OrcaNet** — plus a shared infrastructure layer used by all three.

---

## Components

| Component | Codename | Role |
|-----------|----------|------|
| **OrcaMind** | The Brain | Meta-learning engine: task embedding, model selection, MAML/Reptile/Meta-SGD, warm-start transfer |
| **OrcaLab** | The Lab | Experiment management hub: adaptive hyperparameter search, Prefect orchestration, live dashboards |
| **OrcaNet** | The Connector | Cross-domain knowledge transfer: domain-invariant embeddings, LLM-powered reasoning, transfer scoring |
| **orca-shared** | The Foundation | Shared schemas, SQLAlchemy ORM, storage backends, MLflow wrappers, HTTP client library |

---

## Quick Start

The fastest path to a running stack is Docker Compose. OrcaMind, PostgreSQL, Redis, MinIO, and MLflow all start together.

**Prerequisites:** Docker Engine 24+, Docker Compose v2, Python 3.11+, [uv](https://docs.astral.sh/uv/)

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

---

## What's Built

### `orca-shared` — Shared Infrastructure

#### SQLAlchemy ORM (`registry/models.py`)

Seven fully-typed `Mapped[]` models backed by PostgreSQL:

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `tasks` | ML tasks / datasets | `name`, `domain`, `task_type`, `n_samples`, `n_features`, `n_classes`, `metadata` (JSONB) |
| `embeddings` | Task embedding vectors | `embedding_vector` (ARRAY Float), `embedding_type`, `dimension`, `model_version` |
| `models` | Model architectures | `name`, `architecture`, `config` (JSONB), `parameter_count`, `flops` |
| `experiments` | Training runs | `task_id`, `model_id`, `training_config` (JSONB), `status`, `mlflow_run_id` |
| `performances` | Per-run metrics | `metric_name`, `metric_value`, `epoch`, `is_final`, `metadata` (JSONB) |
| `transfer_mappings` | Task-to-task transfer scores | `source_task_id`, `target_task_id`, `transfer_score`, `transfer_type` |
| `search_spaces` | Hyperparameter definitions | `name`, `definition` (JSONB), `parent_id` (self-referential tree) |

`tasks` ↔ `embeddings` have a circular foreign key handled by `use_alter=True` in the ORM and a deferred `op.create_foreign_key` in the Alembic migration.

#### Repository Layer (`registry/repository.py`)

Async repository pattern over all tables:

- `TaskRepository` — `list_all()`, `list_by_domain()`, `list_by_type()`, `get_by_id()`, `create()`, `update_embedding()`
- `ExperimentRepository` — `create()`, `get_by_id()`, `list_by_task()`, `update_status()`, `mark_complete()`
- `PerformanceRepository` — `log_metric()`, `get_final_metrics()`, `get_history()`, `list_all_with_context()`
- `EmbeddingRepository` — `create()`, `get_by_id()`

#### Pydantic v2 Schemas (`schemas/`)

20+ validated models across 7 files:

| File | Models |
|------|--------|
| `task.py` | `TaskCreate`, `Task`, `TaskSummary`, `DatasetSummary` |
| `embedding.py` | `Embedding`, `SimilarityResult` |
| `model.py` | `ModelConfig`, `ModelSummary` |
| `recommendation.py` | `RecommendationRequest`, `ModelRecommendation`, `FeedbackRequest` |
| `training.py` | `TrainingConfig`, `ExperimentResult` |
| `transfer.py` | `TransferMapping`, `TransferScore`, `TransferRecommendation` |
| `metrics.py` | `MetricPoint`, `PerformanceMetrics`, `PerformanceSummary` |

#### Storage Backends (`storage/`)

- `StorageBackend` (ABC): `upload()`, `download()`, `delete()`, `exists()`
- `LocalBackend`: filesystem with path-traversal protection
- `MinIOBackend`: S3-compatible object storage via minio-py

#### Experiment Tracking (`tracking/`)

- `OrcaTracker`: async context manager for MLflow run lifecycle
- `MetricLogger`: batch `mlflow.log_metrics()` wrapper
- `ArtifactManager`: `upload_model()` / `download_model()` with `weights_only=True`
- `ModelRegistry`: stage-based versioning (Staging → Production → Archived)

#### HTTP Clients (`clients/`)

Async `httpx`-based clients for inter-service calls:

- `OrcaMindClient`: `/api/v1/recommend-model`, `/api/v1/predict-performance`, `/api/v1/similar-tasks`
- `OrcaLabClient`: adaptive search calls (stub, planned)
- `OrcaNetClient`: transfer scoring calls (stub, planned)

---

### `orcamind` — Meta-Learning Engine

#### Core Algorithms (`orcamind.core`)

| Module | Algorithm | Reference |
|--------|-----------|-----------|
| `maml.py` | MAML — first- and second-order meta-gradients via `torch.autograd.grad(create_graph=True)` | Finn et al. 2017 |
| `reptile.py` | Reptile — first-order interpolation (Polyak averaging on adapted params) | Nichol et al. 2018 |
| `meta_sgd.py` | Meta-SGD — per-parameter learnable inner LRs clamped to ≥1e-8 | Li et al. 2017 |
| `warmstart.py` | WarmStartTransfer — segment-aware layer matching + fine-tuning schedules | — |
| `base.py` | `MetaLearner` abstract base: `adapt()`, `inner_loop()`, `meta_update()` | — |

#### Task Embedders (`orcamind.embedders`)

- **`StatisticalEmbedder`** — 25-dimensional meta-feature vector: log(samples), log(features), class balance, entropy, skewness, kurtosis, feature correlation, mutual information
- **`NeuralEmbedder`** — MLP over statistical features with contrastive loss; output dim 64
- **`FaissIndex`** — cosine-similarity k-NN over task embeddings: `add()`, `search()`, `save()`, `load()`

#### Model Selectors (`orcamind.selectors`)

- **`NearestNeighborSelector`** — finds *k* most similar past tasks, votes on best-performing model
- **`LearningToRankSelector`** — XGBoost ranker over `(task_embedding, model_config) → performance`
- **`PerformancePredictor`** — estimates final metric ∈ [0, 1] plus confidence; used for selection and early stopping

#### Meta-Training Pipeline (`orcamind.training`)

- **`MetaTrainer`** — PyTorch Lightning module; wraps meta-learner + sampler; logs to MLflow; DDP-compatible
- **`TaskSampler`** — three strategies: uniform random, curriculum (difficulty-aware), domain-balanced
- **`MetaValidationCallback`** / **`EarlyStoppingCallback`** / **`CheckpointCallback`**
- **`MetaMetrics`** — `k_shot_accuracy`, `adaptation_efficiency`, `catastrophic_forgetting`

#### REST API (`orcamind.api`)

12 endpoints served by FastAPI, documented at `GET /docs`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info (name, version, status) |
| `GET` | `/health` | Liveness probe — `{status, db, faiss, mlflow}` booleans |
| `GET` | `/api/v1/tasks` | Paginated task list; filter by `domain` or `task_type` |
| `GET` | `/api/v1/tasks/{task_id}` | Task detail — 404 if not found |
| `POST` | `/api/v1/tasks/embed` | Store a pre-computed task embedding |
| `POST` | `/api/v1/recommend-model` | Top-*k* model recommendations via `NearestNeighborSelector` |
| `POST` | `/api/v1/predict-performance` | Performance estimate + confidence from `PerformancePredictor` |
| `POST` | `/api/v1/similar-tasks` | FAISS k-NN lookup → ranked `SimilarityResult` list |
| `POST` | `/api/v1/feedback` | Log final experiment metric; closes the meta-learning loop |
| `GET` | `/api/v1/models` | Available model architectures |
| `POST` | `/api/v1/adapt` | Dispatch async meta-adaptation job — returns `{job_id}` immediately |
| `GET` | `/api/v1/performances` | Mean metrics grouped by (task, architecture) — powers the Performance Heatmap |

**Architecture highlights:**

- `create_app()` factory — all singletons (DB engine, embedder, selectors, FAISS index) initialised once at startup via ASGI lifespan, injected per-request via `Depends()`
- **Graceful degradation** — FAISS index is optional; if absent at boot, `/health` reports `faiss: false` and the service stays up
- **CORS** — allowed origins from `CORS_ORIGINS` env var (comma-separated)
- **Background adaptation** — `POST /adapt` creates an experiment record, fires `_run_adaptation` as a `BackgroundTask`, and returns immediately

#### CLI (`orcamind`)

Full-featured Typer CLI installed as the `orcamind` entry point.

```bash
orcamind --help           # List all commands
orcamind <command> --help # Per-command usage
```

| Command | Purpose | Key Options |
|---------|---------|-------------|
| `init` | Create `data/`, `models/`, `logs/`, `config/config.yaml`; register MLflow experiment | — |
| `train` | MAML meta-training loop | `--config PATH`, `--epochs INT`, `--device cpu\|cuda` |
| `embed` | Compute 25-dim statistical + 64-dim neural task embedding | `--output PATH` |
| `recommend` | Embed dataset → call API → render top-*k* recommendations table | `--top-k INT`, `--api-url URL` |
| `serve` | Start FastAPI via Uvicorn | `--host TEXT`, `--port INT`, `--reload` |
| `dashboard` | Launch Streamlit dashboard | `--port INT` |

`train` and `embed` use lazy imports — if PyTorch is absent the command prints an install hint and exits cleanly.

#### Streamlit Dashboard (`orcamind.dashboard`)

Four-page application launched via `orcamind dashboard` or `streamlit run orcamind/dashboard/app.py`.

| Page | File | What it shows |
|------|------|---------------|
| **Task Browser** | `pages/task_browser.py` | Filterable task table (domain, task type); JSON detail panel; 2-D PCA scatter of meta-features with selection highlight |
| **Training Progress** | `pages/training_progress.py` | MLflow multi-run comparison; epoch-level loss + accuracy line charts; optional 30s auto-refresh |
| **Recommendation Explorer** | `pages/recommendation_explorer.py` | CSV upload → statistical embedding → top-3 recommendation cards from `/recommend-model` → similar-task similarity bar chart from `/similar-tasks` |
| **Performance Heatmap** | `pages/performance_heatmap.py` | Task × architecture accuracy matrix from `/performances` — interactive RdYlGn Plotly heatmap; gray for missing cells; raw data table below |

All pages read the API base URL and MLflow URI from sidebar inputs.

#### Hydra Configuration (`config/`)

```
config/
├── config.yaml       # Root: paths, mlflow_uri, seed, device
├── model/
│   └── maml.yaml     # inner_lr, outer_lr, n_inner_steps, base_model
├── dataset/
│   └── openml.yaml   # suite, max_tasks, output_dir
└── optimizer/
    └── adam.yaml     # lr, weight_decay, betas
```

---

## Database Migrations

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

| Revision | Description |
|----------|-------------|
| `0001` | Initial schema — all 7 registry tables in FK-safe creation order; deferred `fk_tasks_embedding_id` to resolve the `tasks ↔ embeddings` circular dependency |

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

| Flag | Default | Description |
|------|---------|-------------|
| `--suites` | `cc18 ctr23` | OpenML benchmark suites to download |
| `--max-tasks INT` | all | Cap on tasks per suite |
| `--output-dir PATH` | `data/` | Directory for FAISS index output |
| `--db-url URL` | from `DATABASE_URL` | Override database connection |
| `--dry-run` | off | Parse + embed without writing to the DB |

**What it does:**

1. Downloads **OpenML CC-18** (classification, ≤72 tasks) and/or **CTR-23** (regression)
2. For each task: extracts features from the raw dataset and computes a 25-dim statistical embedding
3. Runs 5 baseline models (Logistic Regression, Random Forest, XGBoost, SVM, KNN) with 5-fold cross-validation, skipping SVM/SVR for datasets >10,000 samples
4. Persists `Task`, `Model`, `Experiment`, `Performance` rows to PostgreSQL via the repository layer
5. Adds each task embedding to an in-memory FAISS cosine-similarity index
6. Saves the completed index to `{output-dir}/orca_task_index.faiss`

After seeding, `GET /api/v1/tasks` and the Recommendation Explorer will return real data.

---

## Development

### Running tests

```bash
# All tests
pytest packages/ -v --cov

# Unit tests only (fast, no services required)
pytest packages/orcamind/tests/unit/ -v

# Integration tests (requires docker-compose stack)
pytest packages/orcamind/tests/integration/ -v
```

The test suite has 51 test files across unit and integration categories. Integration tests auto-skip when their target service port is unreachable — run `make docker-up` first to exercise them.

### Linting and type checking

```bash
ruff check .          # Lint
ruff format .         # Format
mypy packages/        # Type check (strict on orca-shared)
```

### Pre-commit hooks

```bash
pip install pre-commit
pre-commit install    # Install on git commit + push hooks
pre-commit run --all-files
```

Hooks run: ruff lint, ruff format, mypy. The push stage runs the unit test suite.

### Makefile targets

```bash
make install      # uv venv + install all packages
make test         # pytest with coverage
make lint         # ruff check
make type-check   # mypy
make docker-up    # docker compose up -d
make docker-down  # docker compose down
make docker-logs  # docker compose logs -f
make clean        # remove __pycache__, .pytest_cache, .mypy_cache
make help         # list all targets
```

---

## Getting Started (Local Dev Mode)

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

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Orca Ecosystem                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │  OrcaMind   │ ←→  │   OrcaLab   │ ←→  │   OrcaNet   │       │
│   │  port 8000  │     │  port 8001  │     │  port 8002  │       │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘       │
│          └──────────────────┬┘────────────────────┘             │
│                             │                                    │
│          ┌──────────────────▼──────────────────┐                │
│          │          orca-shared                 │                │
│          ├─────────────────────────────────────┤                │
│          │  Registry  (PostgreSQL + SQLAlchemy) │                │
│          │  Migrations (Alembic)                │                │
│          │  Artifacts  (MinIO / Local FS)       │                │
│          │  Tracking   (MLflow)                 │                │
│          │  Vector search (FAISS)               │                │
│          │  Schemas   (Pydantic v2)             │                │
│          └─────────────────────────────────────┘                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```text
orca/
├── packages/
│   ├── orca-shared/                  # Shared infrastructure layer
│   │   └── orca_shared/
│   │       ├── clients/              # Async httpx clients (OrcaMind, OrcaLab, OrcaNet)
│   │       ├── registry/             # SQLAlchemy ORM models + async repository layer
│   │       ├── schemas/              # Pydantic v2 data contracts (20+ models)
│   │       ├── storage/              # LocalBackend + MinIOBackend
│   │       └── tracking/             # MLflow wrappers (OrcaTracker, ArtifactManager, ModelRegistry)
│   │
│   └── orcamind/                     # Meta-learning engine
│       ├── orcamind/
│       │   ├── core/                 # MAML, Reptile, Meta-SGD, WarmStartTransfer, base
│       │   ├── embedders/            # StatisticalEmbedder, NeuralEmbedder, FaissIndex
│       │   ├── selectors/            # NearestNeighbor, LearningToRank, PerformancePredictor
│       │   ├── training/             # MetaTrainer, TaskSampler, callbacks, metrics
│       │   ├── api/                  # FastAPI app factory + 12 endpoints across 7 routers
│       │   ├── dashboard/            # Streamlit app (app.py + 4 pages)
│       │   └── cli.py                # Typer CLI — 7 commands
│       ├── alembic/                  # Database migration environment
│       │   ├── env.py                # Async SQLAlchemy migration runner
│       │   ├── script.py.mako        # Revision template
│       │   └── versions/
│       │       └── 0001_initial_schema.py  # All 7 tables
│       ├── alembic.ini               # Alembic configuration
│       ├── scripts/
│       │   └── init_db.py            # Run alembic upgrade head (used by Docker Compose)
│       ├── config/                   # Hydra YAML configs (root, model, dataset, optimizer)
│       └── tests/
│           ├── unit/                 # 40+ unit test files (no services required)
│           └── integration/          # API + Docker service smoke tests
│
├── scripts/
│   └── bootstrap_meta_dataset.py    # Seed registry from OpenML CC-18 / CTR-23
│
├── docker-compose.dev.yml            # Full dev stack: Postgres, Redis, MinIO, MLflow, OrcaMind
├── Makefile                          # install, test, lint, type-check, docker-up/down/logs, clean
├── pyproject.toml                    # uv workspace config + ruff / mypy / pytest settings
└── .pre-commit-config.yaml           # ruff + mypy + unit-test hooks
```

---

## Tech Stack

### ML & Meta-Learning

- **PyTorch 2.0+** + **PyTorch Lightning** for meta-training
- **learn2learn** + **higher** for differentiable inner-loop optimization (MAML second-order)
- **FAISS** for approximate nearest-neighbor search over task embeddings
- **scikit-learn**, **XGBoost**, **SciPy** for selectors and statistical embedders

### Data & Infrastructure

- **PostgreSQL 15** + **SQLAlchemy 2.0** (async, `asyncpg`) for the meta-registry
- **Alembic** for schema migrations
- **MinIO** for S3-compatible artifact storage
- **MLflow 2.10** for experiment tracking and model versioning
- **Redis 7** for caching and event bus
- **OpenML** for benchmark task acquisition

### Configuration & API

- **Hydra / OmegaConf** for hierarchical, composable configuration
- **Pydantic v2** for schema validation across all component boundaries
- **FastAPI** + **Uvicorn** for REST APIs (12 endpoints)
- **Typer** + **Rich** for the CLI
- **Streamlit** + **Plotly** for the analytics dashboard

### Developer Tooling

- **uv** workspace for monorepo package management
- **ruff** for linting and formatting (line length 100, Python 3.11 target)
- **mypy** (strict on `orca-shared`) for static type checking
- **pytest** + **pytest-asyncio** + **pytest-cov** for testing (51 test files, 627+ tests)
- **pre-commit** hooks for quality gates on commit and push

---

## Roadmap

### OrcaMind — Next

- Dataset2Vec neural embedder (end-to-end from raw tabular data)
- Hydra config enhancements for distributed `orcamind train`

### OrcaLab — Planned

- Adaptive hyperparameter search (Optuna with meta-priors from OrcaMind)
- Prefect 2.x orchestration flows: single experiment, sweep, meta-informed sweep, continuous learning
- ASHA pruning (target ≥40% compute reduction vs no pruning)
- Live Streamlit dashboard with WebSocket updates
- Bidirectional OrcaMind ↔ OrcaLab integration (priors in, results out)

### OrcaNet — Planned

- Domain-adversarial cross-domain embedder (DANN)
- Transfer scoring via Centered Kernel Alignment (CKA)
- Hybrid retrieval: FAISS + PostgreSQL metadata filtering + LLM re-ranking
- LangChain reasoning agent for transfer explanations
- Three-way pipeline: OrcaNet → OrcaMind → OrcaLab

### Platform — Planned

- Kubernetes + Helm charts
- GitHub Actions CI/CD (lint, type-check, test, build, push image)
- Prometheus + Grafana monitoring

---

## Reference Papers

| Algorithm | Paper |
|-----------|-------|
| MAML | Finn et al., *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks* (2017) |
| Reptile | Nichol et al., *On First-Order Meta-Learning Algorithms* (2018) |
| Meta-SGD | Li et al., *Meta-SGD: Learning to Learn Quickly for Few-Shot Learning* (2017) |
| Dataset2Vec | Jomaa et al., *Dataset2Vec: Learning Dataset Meta-Features* (2021) |
| CKA | Kornblith et al., *Similarity of Neural Network Representations Revisited* (2019) |
| DANN | Ganin et al., *Domain-Adversarial Training of Neural Networks* (2016) |
| ASHA | Li et al., *Massively Parallel Hyperparameter Tuning* (2018) |

---

*Build the pod. Make it intelligent. Make it work together.*
