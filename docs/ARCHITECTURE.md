# Architecture

> Part of the [Orca](../README.md) meta-learning platform.

---

## System Diagram

```text
┌──────────────────────────────────────────────────────────────────┐
│                         Orca Ecosystem                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│   │  OrcaMind   │ ←→  │   OrcaLab   │ ←→  │   OrcaNet   │        │
│   │  port 8000  │     │  port 8001  │     │  port 8002  │        │
│   └──────┬──────┘     └─────┬───────┘     └──────┬──────┘        │
│          └─────────────────┬┘────────────────────┘               │
│                            │                                     │
│          ┌─────────────────▼────────────────────┐                │
│          │          orca-shared                 │                │
│          ├──────────────────────────────────────┤                │
│          │  Registry  (PostgreSQL + SQLAlchemy) │                │
│          │  Migrations (Alembic)                │                │
│          │  Artifacts  (MinIO / Local FS)       │                │
│          │  Tracking   (MLflow)                 │                │
│          │  Vector search (FAISS)               │                │
│          │  Schemas   (Pydantic v2)             │                │
│          └──────────────────────────────────────┘                │
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
│   ├── orcamind/                     # Meta-learning engine
│   │   ├── orcamind/
│   │   │   ├── core/                 # MAML, Reptile, Meta-SGD, WarmStartTransfer, base
│   │   │   ├── embedders/            # StatisticalEmbedder, NeuralEmbedder, FaissIndex
│   │   │   ├── selectors/            # NearestNeighbor, LearningToRank, PerformancePredictor
│   │   │   ├── training/             # MetaTrainer, TaskSampler, callbacks, metrics
│   │   │   ├── api/                  # FastAPI app factory + 12 endpoints across 7 routers
│   │   │   ├── dashboard/            # Streamlit app (app.py + 4 pages)
│   │   │   └── cli.py                # Typer CLI — 6 commands
│   │   ├── alembic/                  # Database migration environment
│   │   │   ├── env.py                # Async SQLAlchemy migration runner
│   │   │   ├── script.py.mako        # Revision template
│   │   │   └── versions/
│   │   │       └── 0001_initial_schema.py  # All 7 tables
│   │   ├── alembic.ini               # Alembic configuration
│   │   ├── scripts/
│   │   │   └── init_db.py            # Run alembic upgrade head (used by Docker Compose)
│   │   ├── config/                   # Hydra YAML configs (root, model, dataset, optimizer)
│   │   └── tests/
│   │       ├── unit/                 # 40+ unit test files (no services required)
│   │       └── integration/          # API + Docker service smoke tests
│   │
│   └── orcalab/                      # Experiment orchestration hub
│       ├── orcalab/
│       │   ├── experiments/          # Experiment lifecycle (states, runner, batch runner)
│       │   ├── search/               # SearchStrategy ABC, RandomSearch, GridSearch, BayesianSearch, EvolutionarySearch, MetaInformedSearch
│       │   ├── search_spaces/        # Composable, type-safe search space definitions
│       │   ├── pruning/              # ASHA, median, and meta-informed trial pruners
│       │   ├── orchestration/
│       │   │   ├── flows/            # Prefect flows (single experiment, sweep, meta sweep)
│       │   │   └── tasks/            # Prefect tasks (prepare, train, evaluate, log)
│       │   ├── visualization/        # Streamlit dashboard (live experiments, search progress, results)
│       │   ├── api/                  # FastAPI app + WebSocket streaming
│       │   └── cli.py                # Typer CLI — 4 commands
│       ├── config/                   # Hydra YAML configs (root, search/bayesian, pruner/asha)
│       └── tests/
│           ├── unit/
│           │   ├── search/           # SearchStrategy, RandomSearch, GridSearch, BayesianSearch, EvolutionarySearch — 78+ tests
│           │   ├── search_spaces/    # Parameter types, SearchSpace sampling/serialization, SearchSpaceComposer — 44 tests
│           │   └── *.py              # Package import, metadata, CLI, and config tests
│           └── integration/          # API + sweep lifecycle tests
│
├── scripts/
│   └── bootstrap_meta_dataset.py    # Seed registry from OpenML CC-18 / CTR-23
│
├── docker-compose.dev.yml            # Full dev stack: Postgres, Redis, MinIO, MLflow, Prefect, OrcaMind, OrcaLab
├── Makefile                          # install, test, lint, type-check, docker-up/down/logs, clean
├── pyproject.toml                    # uv workspace config + ruff / mypy / pytest settings
└── .pre-commit-config.yaml           # ruff + mypy + unit-test hooks
```

> See [Components](COMPONENTS.md) for implementation details of each layer.

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
- **FastAPI** + **Uvicorn** for REST APIs; **WebSockets** for real-time metric streaming
- **Typer** + **Rich** for the CLI
- **Streamlit** + **Plotly** for the analytics dashboard

### Experiment Orchestration (OrcaLab)

- **Optuna** for Bayesian (TPE), random, and grid hyperparameter search
- **CMA-ES** (`cma`) for evolutionary search with covariance matrix adaptation
- **Prefect 2.x** for experiment orchestration flows and work-pool scheduling

### Developer Tooling

- **uv** workspace for monorepo package management
- **ruff** for linting and formatting (line length 100, Python 3.11 target)
- **mypy** (strict on `orca-shared`) for static type checking
- **pytest** + **pytest-asyncio** + **pytest-cov** for testing (56+ test files across all packages)
- **pre-commit** hooks for quality gates on commit and push
