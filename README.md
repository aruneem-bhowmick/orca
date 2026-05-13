**🐋 Orca**

**A unified meta-learning platform. Teach machines how to learn, not just what to learn.**

---

Orca is a monorepo meta-learning ecosystem built around one idea: **prior experiments are a dataset, and we should learn from them**. Rather than starting every new ML task from scratch, Orca accumulates knowledge across tasks, embedding what it has seen and using that memory to recommend models, warm-start training, and guide hyperparameter search.

The ecosystem is composed of three interconnected services — **OrcaMind**, **OrcaLab**, and **OrcaNet** — plus a shared infrastructure layer used by all three.

---

## Components

| Component       | Codename       | Role                                                                                                  |
| --------------- | -------------- | ----------------------------------------------------------------------------------------------------- |
| **OrcaMind**    | The Brain      | Meta-learning engine: task embedding, model selection, MAML/Reptile/Meta-SGD, warm-start transfer     |
| **OrcaLab**     | The Lab        | Experiment management hub: adaptive hyperparameter search, Prefect orchestration, live dashboards     |
| **OrcaNet**     | The Connector  | Cross-domain knowledge transfer: domain-invariant embeddings, LLM-powered reasoning, transfer scoring |
| **orca-shared** | The Foundation | Shared schemas, SQLAlchemy ORM, storage backends, MLflow wrappers, HTTP client library                |

---

## What's Built

### `orca-shared` — Shared Infrastructure

- **SQLAlchemy 2.0 ORM**: fully typed `Mapped[]` models for `Task`, `Embedding`, `Model`, `Experiment`, `Performance`, `TransferMapping`, `SearchSpace`
- **Repository pattern**: `TaskRepository`, `EmbeddingRepository`, `ExperimentRepository`, `PerformanceRepository` with async PostgreSQL support
- **Storage abstraction**: pluggable `StorageBackend` with `LocalBackend` and `MinIOBackend` implementations
- **MLflow wrappers**: `OrcaTracker`, `MetricLogger`, `ArtifactManager`, `ModelRegistry`
- **Pydantic v2 schemas**: `Task`, `Embedding`, `PerformanceMetrics`, `TrainingConfig`, `TransferMapping`, `RecommendationResult`, and more
- **HTTP client library**: `OrcaMindClient`, `OrcaLabClient`, `OrcaNetClient` built on `httpx`

### `orcamind` — Meta-Learning Engine

#### Core Algorithms (`orcamind.core`)

| Module         | Algorithm                                                                | Reference          |
| -------------- | ------------------------------------------------------------------------ | ------------------ |
| `maml.py`      | MAML — Model-Agnostic Meta-Learning (first- and second-order)            | Finn et al. 2017   |
| `reptile.py`   | Reptile — first-order meta-learning via interpolation                    | Nichol et al. 2018 |
| `meta_sgd.py`  | Meta-SGD — per-parameter learnable inner learning rates                  | Li et al. 2017     |
| `warmstart.py` | Warm-Start Transfer — selective layer transfer and fine-tuning schedules | —                  |
| `base.py`      | `MetaLearner` abstract base with `inner_loop`, `meta_update`, `adapt`    | —                  |

#### Task Embedders (`orcamind.embedders`)

- `StatisticalEmbedder`: extracts a 25-dimensional meta-feature vector from any tabular dataset (log-sample count, dimensionality, class balance, entropy, skewness, kurtosis, feature correlation, mutual information)
- `NeuralEmbedder`: MLP that maps statistical features to a learned compact embedding
- `FaissIndex`: cosine-similarity search over task embeddings — add, search, save, load

#### Model Selectors (`orcamind.selectors`)

- `NearestNeighborSelector`: finds *k* most similar tasks in the registry and votes on the best-performing model
- `LearningToRankSelector`: learns a ranker over `(task_embedding, model_config) → performance`
- `PerformancePredictor`: predicts final metric given a task and model config — used for selection and confidence estimation

#### Meta-Training (`orcamind.training`)

- `MetaTrainer`: PyTorch Lightning module that wraps the meta-training loop, logs metrics to MLflow, and supports distributed data-parallel training
- `TaskSampler`: three strategies — uniform random, difficulty-aware curriculum, domain-balanced
- `MetaValidationCallback` / early-stopping callback
- `MetaMetrics`: k-shot accuracy, adaptation efficiency, forgetting metrics

#### REST API (`orcamind.api`)

OrcaMind exposes a production-ready **FastAPI** service documented at `GET /docs`.

| Method | Path                          | Description                                                                                              |
| ------ | ----------------------------- | -------------------------------------------------------------------------------------------------------- |
| `GET`  | `/`                           | Service info (name, version, status)                                                                     |
| `GET`  | `/health`                     | Liveness probe — returns `healthy` or `degraded` with per-component booleans (`db`, `faiss`, `mlflow`)  |
| `GET`  | `/api/v1/tasks`               | Paginated task list; filterable by `domain` or `task_type`                                               |
| `GET`  | `/api/v1/tasks/{task_id}`     | Task detail — 404 if not found                                                                           |
| `POST` | `/api/v1/tasks/embed`         | Store a pre-computed embedding for a task                                                                |
| `POST` | `/api/v1/recommend-model`     | Top-*k* model recommendations via `NearestNeighborSelector`                                              |
| `POST` | `/api/v1/predict-performance` | Point estimate + confidence from `PerformancePredictor`                                                  |
| `POST` | `/api/v1/similar-tasks`       | FAISS k-NN lookup → ranked `SimilarityResult` list                                                       |
| `POST` | `/api/v1/feedback`            | Log final experiment metric; closes the meta-learning loop                                               |
| `GET`  | `/api/v1/models`              | Available model architectures                                                                            |
| `POST` | `/api/v1/adapt`               | Dispatch an async meta-adaptation job; returns `job_id`                                                  |

**Architecture highlights:**

- `create_app()` factory — instantiates FastAPI with ASGI lifespan; all singletons (DB engine, embedder, selectors, FAISS index) are initialised once at startup and read per-request via `Depends()`
- **Graceful degradation** — if the FAISS index file is absent at boot, `faiss_index = None` and `/health` reports `faiss: false`; the service stays up for endpoints that don't require it
- **CORS** — allowed origins read from `CORS_ORIGINS` env var (comma-separated); wildcards use `allow_credentials=False` to comply with the CORS spec
- **Background adaptation** — `POST /api/v1/adapt` creates an experiment record, fires `_run_adaptation` as a Starlette `BackgroundTask`, and immediately returns `{"job_id": "..."}` so callers are not blocked

#### CLI (`orcamind`)

OrcaMind ships a full-featured **Typer** CLI installed as the `orcamind` entry point.

```bash
orcamind --version        # Print version and exit
orcamind --help           # List all commands
orcamind <command> --help # Per-command usage
```

##### `orcamind init`

Initialise a workspace: creates `data/`, `models/`, and `logs/` directories, writes a default `config/config.yaml` with Hydra/OmegaConf defaults, and registers the MLflow experiment.

```bash
orcamind init
```

##### `orcamind train`

Launch a MAML meta-training run using the configured meta-learner and task sampler. Saves a checkpoint to `models/orcamind_final.pt` on completion.

```bash
orcamind train [OPTIONS]
```

| Flag             | Short | Default              | Description                              |
| ---------------- | ----- | -------------------- | ---------------------------------------- |
| `--config PATH`  | `-c`  | `config/config.yaml` | Path to OmegaConf/Hydra config file      |
| `--epochs INT`   | `-e`  | `10`                 | Number of meta-training epochs           |
| `--device CHOICE`| `-d`  | `cpu`                | Training device — `cpu` or `cuda`        |

Requires the `orcamind[train]` extras (`torch`, `pytorch-lightning`, `learn2learn`). If training dependencies are absent, the command prints an install hint and exits cleanly.

##### `orcamind embed`

Compute a task embedding for a CSV or Parquet dataset using both `StatisticalEmbedder` (25-dim) and `NeuralEmbedder` (64-dim). Prints JSON to stdout; optionally saves to a file.

```bash
orcamind embed DATASET_PATH [OPTIONS]
```

| Flag              | Short | Default | Description                      |
| ----------------- | ----- | ------- | -------------------------------- |
| `--output PATH`   | `-o`  | —       | Write embedding JSON to this file |

Example output:

```json
{
  "dataset": "data/my_task.csv",
  "statistical": [0.0, 1.3, ...],
  "neural": [0.12, -0.04, ...]
}
```

##### `orcamind recommend`

Embed a dataset and call the OrcaMind REST API to retrieve the top-*k* model recommendations. Renders results as a formatted Rich table in the terminal.

```bash
orcamind recommend DATASET_PATH [OPTIONS]
```

| Flag              | Short | Default                  | Description                           |
| ----------------- | ----- | ------------------------ | ------------------------------------- |
| `--top-k INT`     | `-k`  | `3`                      | Number of model recommendations       |
| `--api-url URL`   | —     | `http://localhost:8000`  | OrcaMind API base URL                 |

##### `orcamind serve`

Start the OrcaMind FastAPI service via Uvicorn using the `create_app` factory.

```bash
orcamind serve [OPTIONS]
```

| Flag           | Short | Default       | Description                                  |
| -------------- | ----- | ------------- | -------------------------------------------- |
| `--host TEXT`  | —     | `127.0.0.1`   | Bind host (pass `0.0.0.0` to expose on LAN)  |
| `--port INT`   | `-p`  | `8000`        | Bind port                                    |
| `--reload`     | —     | off           | Enable auto-reload for development           |

##### `orcamind dashboard`

Launch the OrcaMind Streamlit dashboard for task similarity exploration and recommendation visualisation.

```bash
orcamind dashboard [OPTIONS]
```

| Flag          | Short | Default | Description          |
| ------------- | ----- | ------- | -------------------- |
| `--port INT`  | `-p`  | `8501`  | Streamlit server port |

### Bootstrap Script (`scripts/bootstrap_meta_dataset.py`)

Seeds the registry from OpenML benchmark suites:

```bash
python scripts/bootstrap_meta_dataset.py \
  --suites cc18 ctr23 \
  --max-tasks 20 \
  --output-dir data/ \
  --dry-run
```

- Downloads **OpenML CC-18** (classification, ~72 tasks) and **CTR-23** (regression)
- Runs 5 baseline models (Logistic Regression, Random Forest, XGBoost, SVM, KNN) with 5-fold CV on each task
- Persists `Task`, `Model`, `Experiment`, `Performance` rows to PostgreSQL via the repository layer
- Generates 25-dim statistical embeddings for every task
- Builds and saves a FAISS similarity index to disk

---

## Getting Started

### 1. Clone and Install

```bash
git clone https://github.com/AruneemB/orca.git
cd orca

# Create virtualenv and install all workspace packages
uv venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install -e packages/orca-shared
uv pip install -e packages/orcamind
```

### 2. Start Infrastructure

```bash
docker compose -f docker-compose.dev.yml up -d
# Starts: PostgreSQL, Redis, MinIO, MLflow
```

### 3. Initialise a Workspace

```bash
orcamind init
# Creates data/, models/, logs/, config/config.yaml
# Registers the MLflow experiment
```

### 4. Bootstrap the Meta-Dataset

```bash
export ORCA_DB_URL="postgresql+asyncpg://orca:orca@localhost:5432/orca"

python scripts/bootstrap_meta_dataset.py \
  --suites cc18 ctr23 \
  --output-dir data/
```

### 5. Run the OrcaMind API

```bash
export DATABASE_URL="postgresql+asyncpg://orca:orca@localhost:5432/orca_registry"
export FAISS_INDEX_PATH="data/faiss_index"

orcamind serve
# Interactive docs: http://localhost:8000/docs
# Health probe:     http://localhost:8000/health
```

To expose the service on your local network:

```bash
orcamind serve --host 0.0.0.0 --port 8000
```

### 6. Get Model Recommendations

```bash
# Compute and inspect a task embedding
orcamind embed data/my_task.csv --output embeddings/my_task.json

# Get top-3 model recommendations
orcamind recommend data/my_task.csv --top-k 3
```

### 7. Run a Meta-Training Cycle

```bash
orcamind train --config config/config.yaml --epochs 20 --device cpu
```

### 8. Run Tests

```bash
pytest packages/ -v --cov
```

### 9. Lint and Type-Check

```bash
ruff check .
mypy packages/
```

---

## Coming Next

### OrcaMind — In Progress

- Hydra config enhancements for advanced `orcamind train` features
- Dataset2Vec neural embedder (end-to-end from raw data)
- Streamlit dashboard content — task similarity exploration and recommendation visualisation
- Docker image for one-command deployment

### OrcaLab — Planned

- Adaptive hyperparameter search (Optuna with meta-priors from OrcaMind)
- Prefect 2.x orchestration flows: single experiment, sweep, meta-informed sweep, continuous learning
- ASHA pruning to cut compute by ≥40% vs no pruning
- Live Streamlit dashboard with WebSocket updates
- Bidirectional OrcaMind ↔ OrcaLab integration (priors in, results out)

### OrcaNet — Planned

- Domain-adversarial cross-domain embedder (DANN)
- Transfer scoring via Centered Kernel Alignment (CKA)
- Hybrid retrieval (FAISS + PostgreSQL metadata filtering + LLM re-ranking)
- LangChain-based reasoning agent for transfer explanations
- Three-way integration: OrcaNet → OrcaMind → OrcaLab pipeline

### Platform

- Docker Compose full-stack deployment
- Kubernetes + Helm charts
- GitHub Actions CI/CD (lint, type-check, test, build, deploy)
- Prometheus + Grafana monitoring

---

## Tech Stack

### ML & Meta-Learning

- **PyTorch 2.0+** + **PyTorch Lightning** for training
- **learn2learn** + **higher** for MAML differentiable optimization
- **FAISS** for approximate nearest-neighbor search over task embeddings
- **scikit-learn**, **XGBoost**, **SciPy** for selectors and statistical embedders

### Data & Infrastructure

- **PostgreSQL** + **SQLAlchemy 2.0** (async) for the meta-registry
- **MinIO / S3** for artifact storage
- **MLflow** for experiment tracking and model versioning
- **OpenML** for benchmark task acquisition
- **Redis** (planned) for caching and event bus

### Configuration & API

- **Hydra / OmegaConf** for hierarchical, composable configuration
- **Pydantic v2** for schema validation across component boundaries
- **FastAPI** + **Uvicorn** for REST APIs (11 endpoints live)
- **Typer** + **Rich** for the CLI

### Developer Tooling

- **uv** workspace for monorepo package management
- **ruff** for linting and formatting
- **mypy** (strict on `orca-shared`) for type checking
- **pytest** + **pytest-asyncio** + **pytest-cov** for testing
- **pre-commit** hooks for quality gates

---

## Repository Structure

```text
orca/
├── packages/
│   ├── orca-shared/             # Shared schemas, ORM, storage, tracking, clients
│   │   └── orca_shared/
│   │       ├── clients/         # HTTP clients for inter-service calls
│   │       ├── registry/        # SQLAlchemy models + repository layer
│   │       ├── schemas/         # Pydantic v2 data contracts
│   │       ├── storage/         # Local + MinIO storage backends
│   │       └── tracking/        # MLflow wrappers
│   │
│   └── orcamind/                # Meta-learning engine
│       ├── orcamind/
│       │   ├── core/            # MAML, Reptile, Meta-SGD, WarmStart
│       │   ├── embedders/       # Statistical, Neural, FAISS similarity
│       │   ├── selectors/       # KNN, ranker, performance predictor
│       │   ├── training/        # Lightning trainer, samplers, callbacks, metrics
│       │   ├── api/             # FastAPI service — 11 REST endpoints
│       │   ├── dashboard/       # Streamlit dashboard application
│       │   └── cli.py           # Typer CLI — init, train, embed, recommend, serve, dashboard
│       └── config/              # Hydra YAML configs
│
├── scripts/
│   └── bootstrap_meta_dataset.py  # Seed registry from OpenML benchmarks
│
├── docker-compose.dev.yml       # Local dev stack (Postgres, Redis, MinIO, MLflow)
├── pyproject.toml               # Root uv workspace config + ruff/mypy/pytest
└── .pre-commit-config.yaml      # Pre-commit hooks
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Orca Ecosystem                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     │
│    │  OrcaMind   │ ←→  │   OrcaLab   │ ←→  │   OrcaNet   │     │
│    │  (Brain)    │     │    (Lab)    │     │    (Net)    │     │
│    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘     │
│           └───────────────────┼───────────────────┘            │
│                               │                                 │
│           ┌───────────────────▼───────────────────┐            │
│           │     orca-shared (Foundation)          │            │
│           ├───────────────────────────────────────┤            │
│           │  Registry (PostgreSQL + SQLAlchemy)   │            │
│           │  Artifact Storage (MinIO / Local)     │            │
│           │  Experiment Tracking (MLflow)         │            │
│           │  Vector Search (FAISS)                │            │
│           │  Shared Schemas (Pydantic v2)         │            │
│           └───────────────────────────────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Reference Papers

| Algorithm   | Paper                                                                                   |
| ----------- | --------------------------------------------------------------------------------------- |
| MAML        | Finn et al., *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks* (2017) |
| Reptile     | Nichol et al., *On First-Order Meta-Learning Algorithms* (2018)                         |
| Meta-SGD    | Li et al., *Meta-SGD: Learning to Learn Quickly for Few-Shot Learning* (2017)           |
| Dataset2Vec | Jomaa et al., *Dataset2Vec: Learning Dataset Meta-Features* (2021)                      |
| CKA         | Kornblith et al., *Similarity of Neural Network Representations Revisited* (2019)       |
| DANN        | Ganin et al., *Domain-Adversarial Training of Neural Networks* (2016)                   |
| ASHA        | Li et al., *Massively Parallel Hyperparameter Tuning* (2018)                            |

---

*Build the pod. Make it intelligent. Make it work together.*
