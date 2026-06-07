# OrcaLab

> The experiment orchestration hub of the [Orca](../../README.md) platform. Codename: The Lab.

---

OrcaLab handles the experiment lifecycle: define hyperparameter search spaces, run adaptive trials, and prune unpromising candidates early. It closes the meta-learning loop with [OrcaMind](../orcamind/README.md) by pulling model priors before sweeps and pushing trial results back after each run. [OrcaNet](../orcanet/README.md) uses it as a validation backend for transfer recommendations.

## Layout

```text
orcalab/
├── experiments/       Trial lifecycle (6-state machine, runner, batch runner)
├── search/            Hyperparameter search strategies (5 concrete + ABC)
├── search_spaces/     Composable, type-safe parameter definitions
├── pruning/           Trial early-stopping strategies (3 concrete + ABC)
├── orchestration/     Prefect 2.x flows and tasks
│   ├── flows/         Single experiment, sweep, meta-sweep, continuous learning
│   └── tasks/         Data prep, training, evaluation, result logging, prior fetching
├── visualization/     Streamlit dashboard, 4 pages + reusable chart components (port 8502)
├── api/               FastAPI REST service, 10 REST + 1 WebSocket endpoint (port 8001)
├── cli.py             Typer CLI: serve, dashboard, config, version
└── config/            Hydra YAML configs (search, pruner)
```

### Experiment Lifecycle

A six-state machine governs every trial:

```text
PENDING → QUEUED → RUNNING → COMPLETED
                          ↘ FAILED
                          ↘ CANCELLED
```

`ExperimentLifecycle` enforces valid transitions with UTC-timestamped audit logging and DB-first persistence ordering (in-memory state updates only after the repository write succeeds). `ExperimentRunner` executes individual trials with per-epoch MLflow metric streaming, pruner integration, configurable retries and timeouts. `BatchExperimentRunner` adds concurrency control via `asyncio.Semaphore`.

### Search Strategies

Six strategies, all implementing the `SearchStrategy` ABC (`suggest` / `update` / `get_best` / `n_trials`):

| Strategy | Engine | Description |
|----------|--------|-------------|
| RandomSearch | Optuna | Seeded random sampling with FIFO pending-trial bookkeeping |
| GridSearch | — | Lazy Cartesian-product grid with per-type discretisation |
| BayesianSearch | Optuna TPE | Tree-Parzen Estimator with prior injection and SQLite/PostgreSQL persistence |
| EvolutionarySearch | CMA-ES | Covariance matrix adaptation with normalised encoding and convergence restart |
| MetaInformedSearch | Optuna TPE | BayesianSearch warm-started with OrcaMind model priors |

### Search Spaces

Composable parameter definitions built from five types (`IntParameter`, `FloatParameter`, `LogUniformParameter`, `DiscreteUniformParameter`, `CategoricalParameter`). `SearchSpace` handles conditional sampling and JSON serialisation. `SearchSpaceComposer` supports `merge`, `inherit`, and `restrict` operations for building search spaces from reusable parts.

### Pruning

Three pruning strategies, each implementing the `Pruner` ABC:

| Pruner | Description |
|--------|-------------|
| MedianStoppingPruner | Stops trials performing below the peer-set median |
| ASHAPruner | Successive halving with rung-based promotion (Li et al. 2018); >93% compute savings in simulation |
| MetaPruner | OrcaMind performance prediction as an early-stopping signal; falls back to base pruner on network error |

### Orchestration

Four Prefect 2.x flows compose the runner, search strategies, pruners, storage, and OrcaMind into schedulable work-pool deployments:

| Flow | Purpose |
|------|---------|
| `run_single_experiment` | Execute one trial end-to-end |
| `run_sweep` | Hyperparameter sweep with configurable strategy and pruner |
| `meta_informed_sweep` | Sweep warm-started with OrcaMind priors |
| `continuous_learning_loop` | Outer loop with per-iteration sleep for ongoing optimisation |

Five Prefect tasks handle the inner workload: `prepare_data`, `train_model`, `evaluate`, `log_results`, and `get_orcamind_priors`.

## API

FastAPI service on port 8001 with 10 REST endpoints and 1 WebSocket endpoint:

| Router | Key Endpoints |
|--------|--------------|
| Experiments | CRUD, cancel with atomic optimistic concurrency, WebSocket `/live` stream |
| Sweeps | Trigger Prefect flows, poll status, retrieve objective-sorted results |
| Search Spaces | Persist and list search space definitions |

Interactive docs at `http://localhost:8001/docs`. Full spec in [API Reference](../../docs/API-REFERENCE.md).

## Dashboard

Streamlit application (port 8502) with four pages and three reusable Plotly components:

Pages:
- Live Experiments: auto-refreshing experiment table with per-epoch progress bars and live loss curves.
- Search Progress: sweep parallel coordinates, best-trial sidebar, cumulative trial count.
- Results Explorer: completed-experiment filtering with A/B config diff and metric comparison.
- Meta-Analysis: domain-by-architecture heatmap, task-complexity scatter, cumulative best-metric trends.

Components: `metric_plots` (loss curves, metric bars), `parallel_coords` (hyperparameter visualisation), `pareto_frontier` (two-objective Pareto frontier scatter).

## CLI

```bash
orcalab serve            # Start the FastAPI server (--reload for dev)
orcalab dashboard        # Launch the Streamlit dashboard
orcalab config           # Display current configuration
orcalab version          # Print package version
```

## Configuration

Hydra YAML configs under `config/`:

```text
config/
├── config.yaml            Root config (Prefect, OrcaMind, resource limits)
├── search/bayesian.yaml   Bayesian search parameters
└── pruner/asha.yaml       ASHA pruner parameters
```

## Integration Points

| Direction | Mechanism |
|-----------|-----------|
| OrcaMind → OrcaLab | `get_orcamind_priors` Prefect task fetches embeddings and recommendations before sweeps |
| OrcaLab → OrcaMind | `log_results` Prefect task submits `FeedbackRequest` after each trial |
| OrcaNet → OrcaLab | Transfer validation dispatched via `OrcaLabClient.create_experiment()` |
| OrcaLab → OrcaNet | Validated metrics written back to `transfer_mappings` |

All inter-service calls degrade gracefully. Sweeps run to completion even when OrcaMind or OrcaNet are unreachable.

Integration diagram in [Architecture](../../docs/ARCHITECTURE.md).

## Testing

```bash
pytest packages/orcalab/tests/unit          # 700+ unit tests
pytest packages/orcalab/tests/integration   # API and bidirectional flow tests
pytest packages/orcalab/tests/performance   # ASHA compute-savings benchmarks
```

The test suite covers search strategies, pruning algorithms, lifecycle state transitions, orchestration flows, Streamlit pages, REST API endpoints, and WebSocket behaviour. Prefect is stubbed in `conftest.py` so orchestration tests run without a live server. More detail in [Development](../../docs/DEVELOPMENT.md).

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Search | Optuna, CMA-ES (`cma`) |
| Orchestration | Prefect 2.x |
| ML | PyTorch, PyTorch Lightning |
| API | FastAPI, Uvicorn, WebSockets |
| Tracking | MLflow |
| Config | Hydra, OmegaConf |
| Dashboard | Streamlit, Plotly |
| CLI | Typer |
| Shared | [orca-shared](../orca-shared/README.md), [orcamind](../orcamind/README.md) |

---

[Back to packages](../README.md)
