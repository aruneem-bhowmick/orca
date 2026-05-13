<p align="center">
  <strong>🐋 Orca</strong>
</p>

<p align="center">
  <strong>A unified meta-learning platform. Teach machines how to learn, not just what to learn.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Lightning-2.0+-792EE5?style=flat-square&logo=lightning&logoColor=white" alt="PyTorch Lightning">
  <img src="https://img.shields.io/badge/MLflow-2.10+-0194E2?style=flat-square&logo=mlflow&logoColor=white" alt="MLflow">
  <img src="https://img.shields.io/badge/FAISS-1.7+-00599C?style=flat-square" alt="FAISS">
  <img src="https://img.shields.io/badge/OpenML-0.14+-F7931E?style=flat-square" alt="OpenML">
  <img src="https://img.shields.io/badge/uv-workspace-DE5FE9?style=flat-square" alt="uv">
</p>

---

Orca is a monorepo meta-learning ecosystem built around one idea: **prior experiments are a dataset, and we should learn from them**. Rather than starting every new ML task from scratch, Orca accumulates knowledge across tasks, embedding what it has seen and using that memory to recommend models, warm-start training, and guide hyperparameter search.

The ecosystem is composed of three interconnected services — **OrcaMind**, **OrcaLab**, and **OrcaNet** — plus a shared infrastructure layer used by all three.

## 🧩 Components

| Component | Codename | Role |
|-----------|----------|------|
| **OrcaMind** | The Brain | Meta-learning engine: task embedding, model selection, MAML/Reptile/Meta-SGD, warm-start transfer |
| **OrcaLab** | The Lab | Experiment management hub: adaptive hyperparameter search, Prefect orchestration, live dashboards |
| **OrcaNet** | The Connector | Cross-domain knowledge transfer: domain-invariant embeddings, LLM-powered reasoning, transfer scoring |
| **orca-shared** | The Foundation | Shared schemas, SQLAlchemy ORM, storage backends, MLflow wrappers, HTTP client library |

---

## ✅ What's Built

### `orca-shared` — Shared Infrastructure

- **SQLAlchemy 2.0 ORM**: fully typed `Mapped[]` models for `Task`, `Embedding`, `Model`, `Experiment`, `Performance`, `TransferMapping`, `SearchSpace`
- **Repository pattern**: `TaskRepository`, `EmbeddingRepository`, `ExperimentRepository`, `PerformanceRepository` with async PostgreSQL support
- **Storage abstraction**: pluggable `StorageBackend` with `LocalBackend` and `MinIOBackend` implementations
- **MLflow wrappers**: `OrcaTracker`, `MetricLogger`, `ArtifactManager`, `ModelRegistry`
- **Pydantic v2 schemas**: `Task`, `Embedding`, `PerformanceMetrics`, `TrainingConfig`, `TransferMapping`, `RecommendationResult`, and more
- **HTTP client library**: `OrcaMindClient`, `OrcaLabClient`, `OrcaNetClient` built on `httpx`

### `orcamind` — Meta-Learning Engine

#### Core Algorithms (`orcamind.core`)

| Module | Algorithm | Reference |
|--------|-----------|-----------|
| `maml.py` | MAML — Model-Agnostic Meta-Learning (first- and second-order) | Finn et al. 2017 |
| `reptile.py` | Reptile — first-order meta-learning via interpolation | Nichol et al. 2018 |
| `meta_sgd.py` | Meta-SGD — per-parameter learnable inner learning rates | Li et al. 2017 |
| `warmstart.py` | Warm-Start Transfer — selective layer transfer and fine-tuning schedules | — |
| `base.py` | `MetaLearner` abstract base with `inner_loop`, `meta_update`, `adapt` | — |

#### Task Embedders (`orcamind.embedders`)

- **`StatisticalEmbedder`**: extracts a 25-dimensional meta-feature vector from any tabular dataset (log-sample count, dimensionality, class balance, entropy, skewness, kurtosis, feature correlation, mutual information)
- **`NeuralEmbedder`**: MLP that maps statistical features to a learned compact embedding
- **`FaissIndex`**: cosine-similarity search over task embeddings — add, search, save, load

#### Model Selectors (`orcamind.selectors`)

- **`NearestNeighborSelector`**: finds *k* most similar tasks in the registry and votes on the best-performing model
- **`LearningToRankSelector`**: learns a ranker over `(task_embedding, model_config) → performance`
- **`PerformancePredictor`**: predicts final metric given a task and model config — used for selection and confidence estimation

#### Meta-Training (`orcamind.training`)

- **`MetaTrainer`**: PyTorch Lightning module that wraps the meta-training loop, logs metrics to MLflow, and supports distributed data-parallel training
- **`TaskSampler`**: three strategies — uniform random, difficulty-aware curriculum, domain-balanced
- **`MetaValidationCallback`** / early-stopping callback
- **`MetaMetrics`**: k-shot accuracy, adaptation efficiency, forgetting metrics

#### REST API (`orcamind.api`)

OrcaMind exposes a production-ready **FastAPI** service documented at `GET /docs`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info (name, version, status) |
| `GET` | `/health` | Liveness probe — returns `healthy` or `degraded` with per-component booleans (`db`, `faiss`, `mlflow`) |
| `GET` | `/api/v1/tasks` | Paginated task list; filterable by `domain` or `task_type` |
| `GET` | `/api/v1/tasks/{task_id}` | Task detail — 404 if not found |
| `POST` | `/api/v1/tasks/embed` | Store a pre-computed embedding for a task |
| `POST` | `/api/v1/recommend-model` | Top-*k* model recommendations via `NearestNeighborSelector` |
| `POST` | `/api/v1/predict-performance` | Point estimate + confidence from `PerformancePredictor` |
| `POST` | `/api/v1/similar-tasks` | FAISS k-NN lookup → ranked `SimilarityResult` list |
| `POST` | `/api/v1/feedback` | Log final experiment metric; closes the meta-learning loop |
| `GET` | `/api/v1/models` | Available model architectures |
| `POST` | `/api/v1/adapt` | Dispatch an async meta-adaptation job; returns `job_id` |

**Architecture highlights:**
- **`create_app()` factory** — instantiates FastAPI with ASGI lifespan; all singletons (DB engine, embedder, selectors, FAISS index) are initialised once at startup and read per-request via `Depends()`
- **Graceful degradation** — if the FAISS index file is absent at boot, `faiss_index = None` and `/health` reports `faiss: false`; the service stays up for endpoints that don't require it
- **CORS** — allowed origins read from `CORS_ORIGINS` env var (comma-separated); wildcards use `allow_credentials=False` to comply with the CORS spec
- **Background adaptation** — `POST /api/v1/adapt` creates an experiment record, fires `_run_adaptation` as a Starlette `BackgroundTask`, and immediately returns `{"job_id": "..."}` so callers are not blocked

To start the service locally:

```bash
export DATABASE_URL="postgresql+asyncpg://orca:orca@localhost:5432/orca_registry"
export FAISS_INDEX_PATH="data/faiss_index"   # optional — omit to boot without FAISS
uvicorn orcamind.api.main:create_app --factory --host 0.0.0.0 --port 8000
```

#### CLI (`orcamind`)

```bash
orcamind train    # Launch a meta-training run (Hydra config)
orcamind serve    # Start the FastAPI service
orcamind embed    # Compute and save a task embedding for a CSV
orcamind recommend  # Recommend top-k models for a dataset
```

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

## 🔜 Coming Next

### OrcaMind — In Progress

- [ ] Hydra config wiring for full `orcamind train` pipeline
- [ ] Dataset2Vec neural embedder (end-to-end from raw data)
- [ ] Streamlit dashboard for task similarity exploration
- [ ] Docker image + `orcamind serve` wired to the new API factory

### OrcaLab — Planned

- [ ] Adaptive hyperparameter search (Optuna with meta-priors from OrcaMind)
- [ ] Prefect 2.x orchestration flows: single experiment, sweep, meta-informed sweep, continuous learning
- [ ] ASHA pruning to cut compute by ≥40% vs no pruning
- [ ] Live Streamlit dashboard with WebSocket updates
- [ ] Bidirectional OrcaMind ↔ OrcaLab integration (priors in, results out)

### OrcaNet — Planned

- [ ] Domain-adversarial cross-domain embedder (DANN)
- [ ] Transfer scoring via Centered Kernel Alignment (CKA)
- [ ] Hybrid retrieval (FAISS + PostgreSQL metadata filtering + LLM re-ranking)
- [ ] LangChain-based reasoning agent for transfer explanations
- [ ] Three-way integration: OrcaNet → OrcaMind → OrcaLab pipeline

### Platform

- [ ] Docker Compose full-stack deployment
- [ ] Kubernetes + Helm charts
- [ ] GitHub Actions CI/CD (lint, type-check, test, build, deploy)
- [ ] Prometheus + Grafana monitoring

---

## 🛠 Tech Stack

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
- **Hydra** for hierarchical, composable configuration
- **Pydantic v2** for schema validation across component boundaries
- **FastAPI** + **Uvicorn** for REST APIs (OrcaMind service live with 11 endpoints)
- **Typer** for the CLI

### Developer Tooling
- **uv** workspace for monorepo package management
- **ruff** for linting and formatting
- **mypy** (strict on `orca-shared`) for type checking
- **pytest** + **pytest-asyncio** + **pytest-cov** for testing
- **pre-commit** hooks for quality gates

---

## 🚦 Getting Started

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

### 3. Bootstrap the Meta-Dataset

```bash
export ORCA_DB_URL="postgresql+asyncpg://orca:orca@localhost:5432/orca"

python scripts/bootstrap_meta_dataset.py \
  --suites cc18 ctr23 \
  --output-dir data/
```

### 4. Run the OrcaMind API

```bash
export DATABASE_URL="postgresql+asyncpg://orca:orca@localhost:5432/orca_registry"
export FAISS_INDEX_PATH="data/faiss_index"
uvicorn orcamind.api.main:create_app --factory --host 0.0.0.0 --port 8000
# Interactive docs: http://localhost:8000/docs
# Health probe:     http://localhost:8000/health
```

### 5. Run Tests

```bash
pytest packages/ -v --cov
```

### 6. Lint and Type-Check

```bash
ruff check .
mypy packages/
```

---

## 📂 Repository Structure

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
│       │   └── cli.py           # Typer CLI
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

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Orca Ecosystem                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │  OrcaMind   │ ←→  │   OrcaLab   │ ←→  │   OrcaNet   │        │
│  │  (Brain) ✅ │     │  (Lab) 🔜   │     │ (Net)  🔜   │        │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘        │
│         └───────────────────┼───────────────────┘                │
│                             │                                     │
│         ┌───────────────────▼───────────────────┐                │
│         │     orca-shared (Foundation) ✅        │                │
│         ├───────────────────────────────────────┤                │
│         │  Registry (PostgreSQL + SQLAlchemy)   │                │
│         │  Artifact Storage (MinIO / Local)     │                │
│         │  Experiment Tracking (MLflow)         │                │
│         │  Vector Search (FAISS)                │                │
│         │  Shared Schemas (Pydantic v2)         │                │
│         └───────────────────────────────────────┘                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📚 Reference Papers

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

<p align="center">
  <i>Build the pod. Make it intelligent. Make it work together.</i>
</p>
