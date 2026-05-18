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

### OrcaMind ↔ OrcaLab Bidirectional Data Flow

The `←→` arrow between OrcaMind and OrcaLab represents an active two-way exchange that closes the meta-learning loop:

| Direction | When | Mechanism |
|---|---|---|
| **OrcaMind → OrcaLab** (priors in) | Before a sweep starts | `get_orcamind_priors` Prefect task embeds the task via `GET /api/v1/tasks/{id}/embedding`, requests a model recommendation via `POST /api/v1/recommend-model`, and passes the result to `MetaInformedSearch.initialize_from_orcamind()` which warm-starts the Bayesian search with prior knowledge |
| **OrcaLab → OrcaMind** (feedback out) | After each trial completes | `log_results` Prefect task submits a `FeedbackRequest` to `POST /api/v1/feedback` carrying the experiment ID, the scalar objective metric, and the hyperparameter configuration — feeding completed-trial signal back into OrcaMind's meta-learning data store |

Both directions are fully resilient: network and HTTP errors (`ConnectError`, `TimeoutException`, `HTTPStatusError`) degrade gracefully — sweeps start without priors and run to completion even when OrcaMind is unreachable.

---

## Repository Structure

```text
orca/
├── packages/
│   ├── orca-shared/                  # Shared infrastructure layer
│   │   └── orca_shared/
│   │       ├── clients/              # Async httpx clients (OrcaMind, OrcaLab, OrcaNet)
│   │       ├── registry/             # SQLAlchemy ORM models + async repository layer
│   │       ├── schemas/              # Pydantic v2 data contracts (21+ models)
│   │       ├── storage/              # LocalBackend + MinIOBackend
│   │       └── tracking/             # MLflow wrappers (OrcaTracker, ArtifactManager, ModelRegistry)
│   │
│   ├── orcamind/                     # Meta-learning engine
│   │   ├── orcamind/
│   │   │   ├── core/                 # MAML, Reptile, Meta-SGD, WarmStartTransfer, base
│   │   │   ├── embedders/            # StatisticalEmbedder, NeuralEmbedder, FaissIndex
│   │   │   ├── selectors/            # NearestNeighbor, LearningToRank, PerformancePredictor
│   │   │   ├── training/             # MetaTrainer, TaskSampler, callbacks, metrics
│   │   │   ├── api/                  # FastAPI app factory + 13 endpoints across 7 routers
│   │   │   ├── dashboard/            # Streamlit app (app.py + 4 pages)
│   │   │   └── cli.py                # Typer CLI — 6 commands
│   │   ├── alembic/                  # Database migration environment
│   │   │   ├── env.py                # Async SQLAlchemy migration runner
│   │   │   ├── script.py.mako        # Revision template
│   │   │   └── versions/
│   │   │       ├── 0001_initial_schema.py  # All 7 tables
│   │   │       └── 0002_add_experiment_metrics_column.py  # experiments.metrics JSONB (per-epoch snapshots)
│   │   ├── alembic.ini               # Alembic configuration
│   │   ├── scripts/
│   │   │   └── init_db.py            # Run alembic upgrade head (used by Docker Compose)
│   │   ├── config/                   # Hydra YAML configs (root, model, dataset, optimizer)
│   │   └── tests/
│   │       ├── unit/                 # 40+ unit test files (no services required)
│   │       └── integration/          # API + Docker service smoke tests
│   │
│   └── orcalab/                      # Experiment orchestration hub (API port 8001, Dashboard port 8502)
│       ├── orcalab/
│       │   ├── experiments/          # Experiment lifecycle (states, runner, batch runner)
│       │   ├── search/               # SearchStrategy ABC, RandomSearch, GridSearch, BayesianSearch, EvolutionarySearch, MetaInformedSearch
│       │   ├── search_spaces/        # Composable, type-safe search space definitions
│       │   ├── pruning/              # ASHA, median, and meta-informed trial pruners
│       │   ├── orchestration/
│       │   │   ├── flows/            # Prefect flows (single experiment, sweep, meta sweep)
│       │   │   └── tasks/            # Prefect tasks (prepare_data, train_model, evaluate, log_results, get_orcamind_priors)
│       │   ├── visualization/        # Streamlit dashboard — app entry point + pages + chart components
│       │   │   ├── app.py            # st.navigation() entry point; sidebar API URL input; 4-page layout
│       │   │   ├── components/       # Reusable Plotly components (metric_plots, parallel_coords, pareto_frontier)
│       │   │   └── pages/            # Dashboard pages (live_experiments, search_progress, results_explorer, meta_analysis)
│       │   ├── api/                  # FastAPI app (10 REST + 1 WebSocket endpoint) — port 8001
│       │   │   ├── main.py           # create_app() factory + module-level app instance (uvicorn entrypoint); ASGI lifespan (DB engine, sweeps dict); health + root endpoints
│       │   │   ├── middleware.py     # RequestLoggingMiddleware (try/finally); CORS deny-by-default
│       │   │   ├── deps.py           # get_db, get_experiment_repo, get_search_space_repo, get_sweeps_store
│       │   │   └── routers/
│       │   │       ├── experiments.py  # CRUD + WebSocket /live stream
│       │   │       ├── sweeps.py       # Prefect flow trigger, status poll, results
│       │   │       └── search_spaces.py  # Persist and list search space definitions
│       │   └── cli.py                # Typer CLI — 4 commands
│       ├── config/                   # Hydra YAML configs (root, search/bayesian, pruner/asha)
│       └── tests/
│           ├── unit/
│           │   ├── experiments/      # ExperimentLifecycle, ExperimentRunner (incl. TestEpochTracking), BatchExperimentRunner — 72 tests
│           │   ├── search/           # SearchStrategy, RandomSearch, GridSearch, BayesianSearch, EvolutionarySearch — 78+ tests
│           │   ├── search_spaces/    # Parameter types, SearchSpace sampling/serialization, SearchSpaceComposer — 44 tests
│           │   ├── pruning/          # Pruner ABC, MedianStoppingPruner, ASHAPruner, MetaPruner — 90 tests
│           │   ├── orchestration/    # Prefect task and flow unit tests — 52 tests (Prefect stub in conftest.py)
│           │   ├── visualization/    # Streamlit component and page unit tests — 115 tests
│           │   │   ├── conftest.py   # Session-scoped _patch_streamlit; saves/restores sys.modules on teardown
│           │   │   ├── components/   # test_metric_plots, test_parallel_coords, test_pareto_frontier
│           │   │   └── pages/        # test_app, test_live_experiments, test_search_progress, test_results_explorer, test_meta_analysis
│           │   └── *.py              # Package import, metadata, CLI, config, and deployment validation tests (Dockerfile structure, docker-compose services, Prefect init, app export) — 45 tests
│           ├── integration/
│           │   ├── api/              # OrcaLab REST API integration tests — 70 tests (all external deps mocked; incl. TestWebSocketSpecAssertions)
│           │   └── (OrcaMind bidirectional flows) # 20 integration tests — respx-mocked OrcaMind HTTP API
│           └── performance/          # ASHA compute-savings benchmarks — 4 tests
│                   ├── conftest.py   # ASGITransport client; pre-populates app.state; dependency_overrides for all repos
│                   ├── test_health.py        # Root + health endpoints (DB ok, Prefect degraded)
│                   ├── test_experiments.py   # CRUD, pagination, cancel semantics, atomic update assertion
│                   ├── test_sweeps.py        # Start sweep, status, results, Prefect mock, validation
│                   ├── test_search_spaces.py # Create and list search space records
│                   └── test_websocket.py     # Direct handler invocation — metrics stream, disconnect, terminal status
│
├── scripts/
│   ├── bootstrap_meta_dataset.py    # Seed registry from OpenML CC-18 / CTR-23
│   └── init_prefect.py              # Create orcalab-pool Prefect work pool for sweep flow deployments
│
├── docker-compose.dev.yml            # Full dev stack: Postgres, Redis, MinIO, MLflow, Prefect, OrcaMind, OrcaLab API (8001), OrcaLab Dashboard (8502)
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
- **pytest** + **pytest-asyncio** + **pytest-cov** for testing (80+ test files across all packages)
- **pre-commit** hooks for quality gates on commit and push
