# Architecture

> Part of the [Orca](../README.md) meta-learning platform.

---

## System Diagram

```text
┌──────────────────────────────────────────────────────────────────┐
│                         Orca Platform                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────┐        │
│   │  Browser / SPA                                      │        │
│   └──────────────────────┬──────────────────────────────┘        │
│                          ↓                                       │
│   ┌─────────────────────────────────────────────────────┐        │
│   │  Orca Web (BFF)  ─  port 8003                       │        │
│   │  /auth  /dashboard  /users  /health                 │        │
│   │  /orcamind  /orcalab  /orcanet  (proxy routers)     │        │
│   └───────┬──────────────┬──────────────┬───────────────┘        │
│           ↓              ↓              ↓                        │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│   │  OrcaMind   │←→│   OrcaLab   │←→│   OrcaNet   │             │
│   │  port 8000  │  │  port 8001  │  │  port 8002  │             │
│   └──────┬──────┘  └─────┬───────┘  └──────┬──────┘             │
│          └───────────────┬┘────────────────┘                     │
│                          │                                       │
│          ┌───────────────▼──────────────────┐                    │
│          │          orca-shared             │                    │
│          ├─────────────────────────────────┤                    │
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

### Orca Web — Backend for Frontend (BFF)

Orca Web sits between the browser and the three backend services, providing a single gateway with JWT-based authentication, session management, dashboard aggregation, and service proxy routers. The proxy routers (`/orcamind`, `/orcalab`, `/orcanet`) forward authenticated requests to the upstream services, injecting an `X-Orca-User-ID` header and logging mutating operations (POST) to the `activity_log` table via the `HistoryRepository`. Connection errors return 502; timeouts (10 s) return 504. All responses mirror the upstream status code and body. The `/health` endpoint probes Postgres, Redis, OrcaMind, OrcaLab, and OrcaNet in parallel and returns `"healthy"` (200) or `"degraded"` (503). All BFF endpoints are served under `root_path="/api/v1"` so the OpenAPI schema reflects the production URL structure.

### OrcaMind ↔ OrcaLab Bidirectional Data Flow

The `←→` arrow between OrcaMind and OrcaLab represents an active two-way exchange that closes the meta-learning loop:

| Direction | When | Mechanism |
|---|---|---|
| **OrcaMind → OrcaLab** (priors in) | Before a sweep starts | `get_orcamind_priors` Prefect task embeds the task via `GET /api/v1/tasks/{id}/embedding`, requests a model recommendation via `POST /api/v1/recommend-model`, and passes the result to `MetaInformedSearch.initialize_from_orcamind()` which warm-starts the Bayesian search with prior knowledge |
| **OrcaLab → OrcaMind** (feedback out) | After each trial completes | `log_results` Prefect task submits a `FeedbackRequest` to `POST /api/v1/feedback` carrying the experiment ID, the scalar objective metric, and the hyperparameter configuration — feeding completed-trial signal back into OrcaMind's meta-learning data store |

Both directions are resilient: network and HTTP errors (`ConnectError`, `TimeoutException`, `HTTPStatusError`) degrade gracefully — sweeps start without priors and run to completion even when OrcaMind is unreachable.

### OrcaNet Three-Way Integration

OrcaNet orchestrates both OrcaMind and OrcaLab to deliver end-to-end knowledge transfer:

| Direction | When | Mechanism |
|---|---|---|
| **OrcaNet → OrcaMind** (source retrieval) | At recommendation time | `OrcaMindClient.get_best_model(source_task_id)` retrieves the best-performing model config for a source task; `OrcaMindClient.recommend_model(target_task_id)` fetches candidate architectures for the target domain |
| **OrcaNet → OrcaLab** (validation dispatch) | After scoring, when `transfer_score > 0.4` | `OrcaLabClient.create_experiment(task_id, model_config, tags)` triggers a validation run using the proposed transfer configuration; `wait_for_completion()` polls until the experiment reaches a terminal state |
| **OrcaLab → OrcaNet** (validation result) | On experiment completion | Validated accuracy from `ExperimentResult.metrics` is written back to the `transfer_mappings` row, closing the loop and making the result available to future queries |

OrcaLab calls are guarded by timeouts and degrade gracefully — if validation times out, the transfer mapping is stored with `experiment_result=None`. OrcaMind failures behave differently: `httpx.ConnectError` or `httpx.TimeoutException` from OrcaMind causes the transfer pipeline to raise `ServiceUnavailableError`, and the API returns HTTP 503.

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
│   ├── orca-web/                      # Backend for Frontend gateway (port 8003)
│   │   ├── orca_web/
│   │   │   ├── auth/                 # JWT tokens, OAuth providers, password hashing
│   │   │   ├── models/               # SQLAlchemy User, UserSession, ActivityLog, UserBookmark
│   │   │   ├── repository/           # UserRepository, SessionRepository, HistoryRepository
│   │   │   ├── services/             # Aggregator (proxies OrcaMind, OrcaLab, OrcaNet)
│   │   │   ├── api/
│   │   │   │   ├── main.py           # create_app() factory + lifespan (DB, httpx, Redis); GET /health
│   │   │   │   ├── deps.py           # get_db, get_current_user, get_aggregator
│   │   │   │   ├── middleware.py     # CORS + RequestLoggingMiddleware
│   │   │   │   ├── proxy_utils.py    # Shared proxy forwarding + activity logging utilities
│   │   │   │   └── routers/          # auth.py, dashboard.py, users.py, orcamind.py, orcalab.py, orcanet.py
│   │   │   └── config.py            # pydantic-settings (database, Redis, JWT, OAuth, upstream URLs)
│   │   └── tests/                    # 188 tests, 98% coverage
│   │
│   └── orcanet/                      # Cross-domain knowledge transfer agent (port 8002)
│       ├── orcanet/
│       │   ├── embeddings/           # CrossDomainEmbedder (DANN, implemented); TextTaskEmbedder (sentence-transformers + stats fusion, implemented); ArchitectureEmbedder (GNN-based, implemented)
│       │   │   ├── __init__.py       # Lazy __getattr__ shim — defers import torch until first access
│       │   │   ├── cross_domain.py   # GradientReversalFunction, GradientReversalLayer, _FeatureMLP, CrossDomainEmbedder
│       │   │   ├── text_features.py  # _AddFusion, _AttentionFusion, TextTaskEmbedder
│       │   │   └── architecture_embedder.py  # ArchitectureGraph, ArchitectureEmbedder
│       │   ├── transfer/             # TransferStrategy ABC, TransferScore dataclass, FeatureTransfer (linear CKA, implemented), WeightTransfer (parameter matching + layer-lr optimizer, implemented), ArchitectureTransfer (graph-embedding similarity + config adaptation, implemented), MultiTaskTransfer + MultiTaskModel (joint training with equal/uncertainty/gradnorm weighting, implemented)
│       │   │   ├── __init__.py       # Public re-exports: ArchitectureTransfer, FeatureTransfer, MultiTaskModel, MultiTaskTransfer, WeightTransfer, TransferScore, TransferStrategy, adapt_architecture, get_optimizer_with_layer_lr, linear_cka
│       │   │   ├── base.py           # TransferStrategy ABC — score_transfer, execute_transfer, get_transfer_metadata
│       │   │   ├── types.py          # TransferScore dataclass — overall, layer_scores, recommended_layers, reasoning
│       │   │   ├── feature_transfer.py  # linear_cka (Kornblith et al. 2019), FeatureTransfer (forward-hook activation collection, depth-weighted CKA scoring, weight patching)
│       │   │   ├── weight_transfer.py   # WeightTransfer (name/shape/both parameter matching, kaiming reinit, deepcopy-based transfer), get_optimizer_with_layer_lr (per-parameter Adam with lr decay), _safe_reinit
│       │   │   ├── architecture_transfer.py  # adapt_architecture (input/output dim adaptation), _build_sequential_from_config (ArchConfig → nn.Sequential), ArchitectureTransfer (OrcaMind source lookup, graph-embedding cosine scoring, middle-layer weight copying)
│       │   │   └── multi_task_transfer.py  # _get_backbone_out_dim, MultiTaskModel (nn.ModuleDict heads, compute_loss, compute_uncertainty_loss with nn.ParameterDict log_sigmas), MultiTaskTransfer (add_task, score_transfer via CrossDomainEmbedder, execute_transfer, update_gradnorm_weights)
│       │   ├── retrieval/            # QueryExpander, LLMRanker, HybridRetriever — three-stage async pipeline (implemented)
│       │   │   └── __init__.py       # Public re-exports: QueryExpander, LLMRanker, HybridRetriever
│       │   ├── reasoning/            # LangChain ReAct agent, Pydantic-validated response models, retry logic (implemented)
│       │   │   └── prompts/          # Transfer explanation, task similarity, architecture recommendation templates
│       │   ├── api/                  # FastAPI service — 8 live endpoints on port 8002
│       │   │   ├── main.py           # create_app() factory; ASGI lifespan initialises all singletons; GET /, GET /health
│       │   │   ├── deps.py           # Depends() providers; X-LLM-Provider header override in get_orcanet_agent()
│       │   │   ├── schemas.py        # TransferScoreRequest, TransferRecommendRequest, RetrieveRequest, EmbedRequest, ExplainRequest
│       │   │   ├── middleware.py     # CORS + RequestLoggingMiddleware
│       │   │   └── routers/          # transfer.py, retrieve.py, explain.py, embed.py
│       │   └── cli.py                # Typer CLI — serve and version commands
│       ├── config/                   # Hydra YAML configs
│       │   ├── config.yaml           # Root: llm, retrieval thresholds, orcamind/orcalab URLs
│       │   ├── retriever/hybrid.yaml # FAISS index path, top-k thresholds, similarity threshold
│       │   ├── embedder/cross_domain.yaml  # DANN dims: input=25, embedding=64, n_domains=10
│       │   └── llm/openai.yaml       # Provider (openai), model (gpt-4-turbo), temperature
│       ├── notebooks/
│       │   └── cross_domain_transfer_demo.ipynb  # Interactive end-to-end pipeline notebook
│       └── tests/
│           ├── unit/
│           │   ├── embeddings/       # CrossDomainEmbedder, GRL, TextTaskEmbedder, and ArchitectureEmbedder unit tests — 76 tests
│           │   ├── transfer/         # linear_cka correctness, FeatureTransfer scoring/guards/metadata/structural, execute_transfer — 37 tests; WeightTransfer scoring (name/shape/both), execute_transfer, optimizer LR groups, guards, metadata — 36 tests; ArchitectureTransfer adapt_architecture, score_transfer, execute_transfer (incl. all activations), metadata, guards — 29 tests; MultiTaskModel forward routing, compute_loss, compute_uncertainty_loss, log_sigma gradient flow — 19 tests; MultiTaskTransfer add_task, score_transfer, execute_transfer, metadata, gradnorm weights, uncertainty convergence — 37 tests (158 total)
│           │   ├── retrieval/        # _parse_list_from_response, QueryExpander.expand — 12 tests; _parse_ranked_list, LLMRanker.rerank — 11 tests; _task_to_feature_vector, _deduplicate_and_sort, HybridRetriever.retrieve, retrieve_with_expanded_queries — 12 tests (35 total)
│           │   └── *.py              # Package structure, CLI smoke tests, config validation — 18 tests
│           ├── integration/          # API integration tests (planned)
│           └── benchmarks/           # Recall@10 retrieval, cross-domain embedding quality, transfer quality — 3 benchmark modules (no services required)
│
├── scripts/
│   ├── bootstrap_meta_dataset.py    # Seed registry from OpenML CC-18 / CTR-23
│   ├── init_prefect.py              # Create orcalab-pool Prefect work pool for sweep flow deployments
│   ├── README.md                    # Script reference and usage guide
│   └── tests/                       # pytest suite for both scripts (55 tests, ≥80% coverage)
│
├── docker-compose.dev.yml            # Full dev stack: Postgres, Redis, MinIO, MLflow, Prefect, OrcaMind (8000), OrcaLab API (8001), OrcaLab Dashboard (8502), OrcaNet (8002)
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
- **Domain-Adversarial Neural Networks (DANN)** (Ganin et al. 2016) for domain-invariant task embeddings in OrcaNet
- **Centered Kernel Alignment (CKA)** (Kornblith et al. 2019) for feature-level transfer scoring in OrcaNet
- **sentence-transformers** (`all-MiniLM-L6-v2`) for natural-language task description embedding

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
- **LangChain** (`langchain`, `langchain-openai`, `langchain-anthropic`) for OrcaNet query expansion, LLM-based candidate re-ranking (`LLMRanker`), and the `OrcaNetAgent` ReAct reasoning agent

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
