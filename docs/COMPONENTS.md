# Components

> Part of the [Orca](../README.md) meta-learning platform.

---

## `orca-shared` — Shared Infrastructure

### SQLAlchemy ORM (`registry/models.py`)

Seven fully-typed `Mapped[]` models backed by PostgreSQL:


| Table               | Purpose                      | Key Columns                                                                               |
| ------------------- | ---------------------------- | ----------------------------------------------------------------------------------------- |
| `tasks`             | ML tasks / datasets          | `name`, `domain`, `task_type`, `n_samples`, `n_features`, `n_classes`, `metadata` (JSONB) |
| `embeddings`        | Task embedding vectors       | `embedding_vector` (ARRAY Float), `embedding_type`, `dimension`, `model_version`          |
| `models`            | Model architectures          | `name`, `architecture`, `config` (JSONB), `parameter_count`, `flops`                      |
| `experiments`       | Training runs                | `task_id`, `model_id`, `training_config` (JSONB), `status`, `mlflow_run_id`, `metrics` (JSONB, nullable) |
| `performances`      | Per-run metrics              | `metric_name`, `metric_value`, `epoch`, `is_final`, `metadata` (JSONB)                    |
| `transfer_mappings` | Task-to-task transfer scores | `source_task_id`, `target_task_id`, `transfer_score`, `transfer_type`                     |
| `search_spaces`     | Hyperparameter definitions   | `name`, `definition` (JSONB), `parent_id` (self-referential tree)                         |


`tasks` ↔ `embeddings` have a circular foreign key handled by `use_alter=True` in the ORM and a deferred `op.create_foreign_key` in the Alembic migration.

### Repository Layer (`registry/repository.py`)

Async repository pattern over all tables:

- `TaskRepository` — `list_all()`, `list_by_domain()`, `list_by_type()`, `get_by_id()`, `create()`, `update_embedding()`, `save_transfer_mapping(source_task_id, target_task_id, transfer_score, transfer_type, metadata) → TransferMapping`
- `ExperimentRepository` — `create()`, `get_by_id()`, `list_by_task()`, `list_all(limit, offset)`, `update_status()`, `update_status_if_current(experiment_id, from_status, to_status) -> bool`, `mark_complete()`, `update_metrics(experiment_id, metrics) -> None`
- `PerformanceRepository` — `log_metric()`, `get_final_metrics()`, `get_history()`, `list_all_with_context()`
- `EmbeddingRepository` — `create()`, `get_by_id()`
- `SearchSpaceRepository` — `create(name, definition)`, `list_all(limit, offset)`

`list_all` on both `ExperimentRepository` and `SearchSpaceRepository` applies `order_by(primary_key)` so that `OFFSET`-based pagination returns consistent results as the table changes. `update_status_if_current` issues an atomic conditional `UPDATE … WHERE status = from_status` and returns `True` when one row was affected, `False` when the status had already changed — enabling optimistic concurrency without a separate SELECT.

`update_metrics(experiment_id, metrics)` merges the supplied dict into the stored `experiments.metrics` JSONB column using a read-modify-write cycle. The SELECT issues `WITH FOR UPDATE` to acquire a row-level lock before reading the current value, preventing concurrent epoch writes from overwriting each other. If the experiment row does not exist the call is a no-op. After updating the in-memory dict it calls `session.flush()` so the change is visible within the current transaction without requiring a full commit.

### Pydantic v2 Schemas (`schemas/`)

20+ validated models across 7 files:


| File                | Models                                                            |
| ------------------- | ----------------------------------------------------------------- |
| `task.py`           | `TaskCreate`, `Task`, `TaskSummary`, `DatasetSummary`             |
| `embedding.py`      | `Embedding`, `SimilarityResult`                                   |
| `model.py`          | `ModelConfig`, `ModelSummary`                                     |
| `recommendation.py` | `RecommendationRequest`, `ModelRecommendation`, `FeedbackRequest` |
| `training.py`       | `TrainingConfig`, `ExperimentResult`                              |
| `search_space.py`   | `SearchSpaceRecord`                                               |
| `transfer.py`       | `TransferMapping`, `TransferScore`, `TransferRecommendation`      |
| `metrics.py`        | `MetricPoint`, `PerformanceMetrics`, `PerformanceSummary`         |


### Storage Backends (`storage/`)

- `StorageBackend` (ABC): `upload()`, `download()`, `delete()`, `exists()`
- `LocalBackend`: filesystem with path-traversal protection
- `MinIOBackend`: S3-compatible object storage via minio-py

### Experiment Tracking (`tracking/`)

- `OrcaTracker`: async context manager for MLflow run lifetime. Calls `mlflow.set_experiment()` and `mlflow.start_run()` on enter; ends the run with `"FINISHED"` or `"FAILED"` status on exit. `self._run` is cleared to `None` in a `finally` block inside `__aexit__` so that `run_id` (see below) always returns `None` outside an active run.
  - `log_params(params)` — `mlflow.log_params()` wrapper
  - `log_metric(name, value, step=None)` — `mlflow.log_metric()` wrapper
  - `log_artifact(local_path)` — `mlflow.log_artifact()` wrapper
  - `run_id: str | None` — read-only property exposing the active MLflow run ID (`self._run.info.run_id`) while inside the context; `None` before enter and after exit
- `MetricLogger`: batch `mlflow.log_metrics()` wrapper
- `ArtifactManager`: `upload_model()` / `download_model()` with `weights_only=True`
- `ModelRegistry`: stage-based versioning (Staging → Production → Archived)

### HTTP Clients (`clients/`)

Async `httpx`-based clients for inter-service calls:

- `OrcaMindClient`: fully-implemented async HTTP client for the OrcaMind meta-learning service. All six methods call `response.raise_for_status()` so callers receive `httpx.HTTPStatusError` on 4xx/5xx:
  - `embed_task(task_id)` — `GET /api/v1/tasks/{task_id}/embedding` → `Embedding`
  - `recommend_model(req)` — `POST /api/v1/recommend-model` → `ModelRecommendation` (first item from list; raises `ValueError` if the response list is empty)
  - `predict_performance(task_embedding, model_id)` — `POST /api/v1/predict-performance` → `PerformanceMetrics`
  - `submit_feedback(req)` — `POST /api/v1/feedback` → `None`
  - `find_similar_tasks(embedding, top_k)` — `POST /api/v1/similar-tasks` → `list[SimilarityResult]`
  - `get_best_model(task_id)` — composes `embed_task` + `recommend_model(top_k=1)` → `ModelSummary`
- `OrcaLabClient`: async HTTP client for the OrcaLab experiment orchestration service. Two methods are fully implemented; two remain as stubs:
  - `create_experiment(task_id, model_config, tags)` — `POST /api/v1/experiments`. Serialises `model_config` (accepts Pydantic models or plain dicts) and forwards `model_id` when present. Returns the `experiment_id` string from the response.
  - `wait_for_completion(experiment_id, timeout=3600, poll_interval=30)` — polls `GET /api/v1/experiments/{id}` until status is in `{COMPLETED, FAILED, CANCELLED}`. Raises `TimeoutError` when the deadline is exceeded; the final poll sleeps at most the remaining time so the error fires promptly.
  - `start_sweep(experiment_id, search_space)` — stub, raises `NotImplementedError`
  - `get_sweep_status(sweep_id)` — stub, raises `NotImplementedError`
- `OrcaNetClient`: transfer scoring calls (stub, planned)

---

## `orcamind` — Meta-Learning Engine

### Core Algorithms (`orcamind.core`)


| Module         | Algorithm                                                                                  | Reference          |
| -------------- | ------------------------------------------------------------------------------------------ | ------------------ |
| `maml.py`      | MAML — first- and second-order meta-gradients via `torch.autograd.grad(create_graph=True)` | Finn et al. 2017   |
| `reptile.py`   | Reptile — first-order interpolation (Polyak averaging on adapted params)                   | Nichol et al. 2018 |
| `meta_sgd.py`  | Meta-SGD — per-parameter learnable inner LRs clamped to ≥1e-8                              | Li et al. 2017     |
| `warmstart.py` | WarmStartTransfer — segment-aware layer matching + fine-tuning schedules                   | —                  |
| `base.py`      | `MetaLearner` abstract base: `adapt()`, `inner_loop()`, `meta_update()`                    | —                  |


### Task Embedders (`orcamind.embedders`)

- `**StatisticalEmbedder**` — 25-dimensional meta-feature vector: log(samples), log(features), class balance, entropy, skewness, kurtosis, feature correlation, mutual information
- `**NeuralEmbedder**` — MLP over statistical features with contrastive loss; output dim 64
- `**FaissIndex**` — cosine-similarity k-NN over task embeddings: `add()`, `search()`, `save()`, `load()`

### Model Selectors (`orcamind.selectors`)

- `**NearestNeighborSelector**` — finds *k* most similar past tasks, votes on best-performing model
- `**LearningToRankSelector**` — XGBoost ranker over `(task_embedding, model_config) → performance`
- `**PerformancePredictor**` — estimates final metric ∈ [0, 1] plus confidence; used for selection and early stopping

### Meta-Training Pipeline (`orcamind.training`)

- `**MetaTrainer**` — PyTorch Lightning module; wraps meta-learner + sampler; logs to MLflow; DDP-compatible
- `**TaskSampler**` — three strategies: uniform random, curriculum (difficulty-aware), domain-balanced
- `**MetaValidationCallback**` / `**EarlyStoppingCallback**` / `**CheckpointCallback**`
- `**MetaMetrics**` — `k_shot_accuracy`, `adaptation_efficiency`, `catastrophic_forgetting`

### REST API (`orcamind.api`)

13 endpoints served by FastAPI, documented at `GET /docs`.


| Method | Path                          | Description                                                                   |
| ------ | ----------------------------- | ----------------------------------------------------------------------------- |
| `GET`  | `/`                           | Service info (name, version, status)                                          |
| `GET`  | `/health`                     | Liveness probe — `{status, db, faiss, mlflow}` booleans                       |
| `GET`  | `/api/v1/tasks`               | Paginated task list; filter by `domain` or `task_type`                        |
| `GET`  | `/api/v1/tasks/{task_id}/embedding` | Task embedding vector — 404 if task has no embedding            |
| `GET`  | `/api/v1/tasks/{task_id}`     | Task detail — 404 if not found                                                |
| `POST` | `/api/v1/tasks/embed`         | Store a pre-computed task embedding                                           |
| `POST` | `/api/v1/recommend-model`     | Top-*k* model recommendations via `NearestNeighborSelector`                   |
| `POST` | `/api/v1/predict-performance` | Performance estimate + confidence from `PerformancePredictor`                 |
| `POST` | `/api/v1/similar-tasks`       | FAISS k-NN lookup → ranked `SimilarityResult` list                            |
| `POST` | `/api/v1/feedback`            | Log final experiment metric; closes the meta-learning loop                    |
| `GET`  | `/api/v1/models`              | Available model architectures                                                 |
| `POST` | `/api/v1/adapt`               | Dispatch async meta-adaptation job — returns `{job_id}` immediately           |
| `GET`  | `/api/v1/performances`        | Mean metrics grouped by (task, architecture) — powers the Performance Heatmap |


**Architecture highlights:**

- `create_app()` factory — all singletons (DB engine, embedder, selectors, FAISS index) initialised once at startup via ASGI lifespan, injected per-request via `Depends()`
- **Graceful degradation** — FAISS index is optional; if absent at boot, `/health` reports `faiss: false` and the service stays up
- **CORS** — allowed origins from `CORS_ORIGINS` env var (comma-separated)
- **Background adaptation** — `POST /adapt` creates an experiment record, fires `_run_adaptation` as a `BackgroundTask`, and returns immediately

### CLI (`orcamind`)

Full-featured Typer CLI installed as the `orcamind` entry point.

```bash
orcamind --help           # List all commands
orcamind <command> --help # Per-command usage
```


| Command     | Purpose                                                                              | Key Options                                          |
| ----------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| `init`      | Create `data/`, `models/`, `logs/`, `config/config.yaml`; register MLflow experiment | —                                                    |
| `train`     | MAML meta-training loop                                                              | `--config PATH`, `--epochs INT`, `--device cpu\|cuda` |
| `embed`     | Compute 25-dim statistical + 64-dim neural task embedding                            | `--output PATH`                                      |
| `recommend` | Embed dataset → call API → render top-*k* recommendations table                      | `--top-k INT`, `--api-url URL`                       |
| `serve`     | Start FastAPI via Uvicorn                                                            | `--host TEXT`, `--port INT`, `--reload`              |
| `dashboard` | Launch Streamlit dashboard                                                           | `--port INT`                                         |


`train` and `embed` use lazy imports — if PyTorch is absent the command prints an install hint and exits cleanly.

### Streamlit Dashboard (`orcamind.dashboard`)

Four-page application launched via `orcamind dashboard` or `streamlit run orcamind/dashboard/app.py`.


| Page                        | File                               | What it shows                                                                                                                                     |
| --------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Task Browser**            | `pages/task_browser.py`            | Filterable task table (domain, task type); JSON detail panel; 2-D PCA scatter of meta-features with selection highlight                           |
| **Training Progress**       | `pages/training_progress.py`       | MLflow multi-run comparison; epoch-level loss + accuracy line charts; optional 30s auto-refresh                                                   |
| **Recommendation Explorer** | `pages/recommendation_explorer.py` | CSV upload → statistical embedding → top-3 recommendation cards from `/recommend-model` → similar-task similarity bar chart from `/similar-tasks` |
| **Performance Heatmap**     | `pages/performance_heatmap.py`     | Task × architecture accuracy matrix from `/performances` — interactive RdYlGn Plotly heatmap; gray for missing cells; raw data table below        |


All pages read the API base URL and MLflow URI from sidebar inputs.

### Hydra Configuration (`config/`)

```text
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

## `orcalab` — Experiment Orchestration Hub

### Package Structure

```text
orcalab/
├── experiments/       # Experiment lifecycle (status state machine, runner, batch runner)
├── search/            # Search strategies (random, grid, Bayesian, evolutionary, meta-informed)
├── search_spaces/     # Composable, type-safe search space definitions
├── pruning/           # ASHA, median stopping, and meta-informed trial pruners
├── orchestration/
│   ├── flows/         # Prefect flows (single experiment, sweep, meta-informed sweep, continuous learning loop)
│   └── tasks/         # Prefect tasks (prepare_data, train_model, evaluate, log_results)
├── visualization/     # Streamlit dashboard — entry point, reusable chart components, and four page modules
│   ├── app.py         # st.navigation() entry point with sidebar API URL input
│   ├── components/    # Reusable Plotly chart builders (metric_plots, parallel_coords, pareto_frontier)
│   └── pages/         # Dashboard pages (live_experiments, search_progress, results_explorer, meta_analysis)
├── api/               # FastAPI application and WebSocket endpoint
└── cli.py             # Typer CLI — 4 commands
```

### Experiment Lifecycle and Execution (`experiments/`)

The experiments package is the execution layer that bridges search strategies and pruning with actual model training. It manages state transitions for each trial, streams metrics to MLflow, retries on failure, and runs multiple experiments concurrently.

All public names are importable from `orcalab.experiments`:

```python
from orcalab.experiments import (
    Experiment,
    ExperimentLifecycle,
    ExperimentRunner,
    ExperimentStatus,
    BatchExperimentRunner,
    InvalidTransitionError,
    TrainableModel,
)
```

#### `Experiment` dataclass (`experiment.py`)

Extends `orca_shared.schemas.ExperimentResult` with three additional fields needed to fully specify and execute a trial.

| Field | Type | Description |
|---|---|---|
| `arch_config` | `dict[str, Any] \| None` | Model architecture and hyperparameter config passed to the `model_factory` |
| `training_config` | `TrainingConfig \| None` | Epochs, learning rate, batch size, optimizer, scheduler |
| `tags` | `dict[str, str] \| None` | Free-form key/value metadata (e.g. sweep ID, experiment notes) |

> `arch_config` is named deliberately to avoid Pydantic v2's reserved `model_config` class attribute. All other `ExperimentResult` fields (`experiment_id`, `task_id`, `model_id`, `status`, `mlflow_run_id`, `started_at`, `completed_at`, `metrics`) are inherited unchanged.

#### State Machine (`lifecycle.py`)

`ExperimentStatus` is a `str` enum with six states. Valid transitions are enforced as a closed set — any other edge raises `InvalidTransitionError`.

```text
PENDING ──► QUEUED ──► RUNNING ──► COMPLETED
   │           │            │
   │           └────────────┤
   └──► CANCELLED ◄─────────┘
                              └──► FAILED
```

| Transition | Trigger |
|---|---|
| `PENDING → QUEUED` | Experiment submitted to the work queue |
| `PENDING → CANCELLED` | Cancelled before reaching the queue |
| `QUEUED → RUNNING` | Picked up by a worker |
| `QUEUED → CANCELLED` | Cancelled after queuing but before a worker picks it up |
| `RUNNING → COMPLETED` | Training finished successfully |
| `RUNNING → FAILED` | Unrecoverable error or pruner decision |
| `RUNNING → CANCELLED` | User-initiated cancellation while running |

`ExperimentLifecycle` manages transitions for a single experiment. It takes an `Experiment` and an `ExperimentRepository` at construction time.

```python
lifecycle = ExperimentLifecycle(experiment, repository)
await lifecycle.transition(ExperimentStatus.RUNNING)
await lifecycle.transition(ExperimentStatus.FAILED, reason="OOM on epoch 7")

for entry in lifecycle.audit_log:
    print(entry)
# {"timestamp": "2025-…", "from": "queued", "to": "running", "reason": ""}
# {"timestamp": "2025-…", "from": "running", "to": "failed", "reason": "OOM on epoch 7"}
```

**`transition(new_status, reason="")`** — async. Validates the edge, then calls `repository.update_status_if_current(experiment_id, current_status, new_status)` — an atomic conditional `UPDATE … WHERE status = current_status`. If the database reports zero rows affected because another process concurrently changed the status, `InvalidTransitionError` is raised with a "Concurrent modification" message and both in-memory state and the audit log remain unchanged — no split-brain.

**`audit_log`** — returns a copy of the internal list. Each entry is a `dict` with keys `timestamp` (ISO-8601 UTC), `from`, `to`, and `reason`.

**`InvalidTransitionError`** — raised synchronously before any I/O when the requested edge is not in the valid set.

#### `ExperimentRunner` (`runner.py`)

Executes a single experiment end-to-end: transitions state, trains the model epoch by epoch, streams metrics to MLflow, integrates with a pruner, retries on failure, uploads the checkpoint on success.

```python
runner = ExperimentRunner(
    tracker=OrcaTracker("my_experiment"),
    artifact_manager=ArtifactManager(storage),
    max_retries=2,      # must be >= 0
    timeout=3600,       # seconds; must be > 0
    model_factory=lambda cfg: MyModel(**cfg),
    repository=experiment_repo,   # optional — enables per-epoch DB writes
)
result = await runner.run(experiment, pruner=asha_pruner)
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tracker` | `OrcaTracker` | required | Async context manager for MLflow run lifetime |
| `artifact_manager` | `ArtifactManager` | required | Handles model serialisation and remote storage |
| `max_retries` | `int` | `2` | Maximum retry attempts after the first failure. Raises `ValueError` if `< 0`. |
| `timeout` | `int` | `3600` | Per-attempt wall-clock timeout in seconds. Raises `ValueError` if `<= 0`. |
| `model_factory` | `Callable[[dict], TrainableModel] \| None` | `None` | Called with `experiment.arch_config` to produce a `TrainableModel`. Omitting it raises `NotImplementedError` at runtime. |
| `repository` | `Any \| None` | `None` → `_NullRepository` | Any object with an `async update_metrics(experiment_id, metrics)` method. When `None`, a no-op `_NullRepository` is used so the runner works without a live database session. Inject a real `ExperimentRepository` to enable per-epoch DB writes that the WebSocket live stream can read. |

**`run(experiment, pruner=None)` execution flow:**

1. Transition experiment to `RUNNING` (caller must leave it in `QUEUED`).
2. For each attempt (up to `max_retries + 1` total):
   - Open an MLflow run via `async with tracker`.
   - Call `model_factory(experiment.arch_config)` to instantiate the model.
   - Log `training_config` params to MLflow.
   - Loop epochs `1 → N` (`training_config.epochs`, default 10):
     - Call `model.train_epoch(epoch) → float` (the primary metric, representing training loss).
     - Log the metric to MLflow under the key `"loss"` with the epoch number as `step`.
     - Write `{"loss": value, "epoch": N}` to the repository via `repository.update_metrics()` so the WebSocket live stream reflects current per-epoch progress. The write is a no-op when no repository is injected.
     - If `pruner.should_prune(trial_id, epoch, metric, history)` returns `True`: transition to `FAILED(reason="pruned")` and return immediately. The checkpoint is **not** uploaded.
   - On successful epoch loop: upload checkpoint via `artifact_manager`, transition to `COMPLETED`, return.
   - On exception or timeout: record the error, try the next attempt.
3. After all attempts exhausted: transition to `FAILED(reason=<last exception>)`, return.

**Retry semantics** — all retry attempts occur while the experiment is in `RUNNING` state. The lifecycle records exactly one `QUEUED → RUNNING` entry and exactly one terminal transition (`RUNNING → COMPLETED` or `RUNNING → FAILED`), regardless of retry count. The audit log is never polluted with intermediate failures.

**Timeout semantics** — each attempt is wrapped in `asyncio.wait_for(timeout=self._timeout)`. A `TimeoutError` is treated identically to any other exception: the attempt is counted as failed, the retry counter increments, and once all attempts are exhausted the experiment transitions to `FAILED`. Artifact upload is never attempted when every attempt times out. The `TestTimeoutBehaviour` class in `tests/unit/experiments/test_runner.py` covers this code path with 5 tests: single-attempt failure, exhaustion after retries, zero-retry fast failure, no artifact upload, and recovery when a retry succeeds.

**`TrainableModel` protocol** — any object with a `train_epoch(epoch: int) -> float` method satisfies this interface. The runner is framework-agnostic: PyTorch, scikit-learn, or a mock all work equally.

#### `BatchExperimentRunner` (`batch_runner.py`)

Runs a list of experiments concurrently, capping the number of simultaneous trials via `asyncio.Semaphore`.

```python
batch_runner = BatchExperimentRunner(runner=runner, max_parallel=4)
results = await batch_runner.run_batch(experiments, pruner=asha_pruner)
# results[i] corresponds to experiments[i] regardless of completion order
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `runner` | `ExperimentRunner` | required | The single-experiment runner to delegate to |
| `max_parallel` | `int` | `4` | Maximum concurrent experiments. Raises `ValueError` if `< 1`. |

**`run_batch(experiments, pruner=None)` guarantees:**

- **Order preserved** — results are written into pre-allocated index slots; `results[i]` always matches `experiments[i]`.
- **Failures isolated** — a failed experiment yields a `FAILED` `ExperimentResult`; it does not cancel sibling tasks or raise.
- **Concurrency bounded** — at most `max_parallel` `run()` coroutines hold the semaphore simultaneously.
- **Empty list** — returns `[]` without touching the runner.

---

### Search Spaces (`search_spaces/`)

Typed, composable hyperparameter definitions that wrap the Optuna trial API. Every downstream search strategy (random, Bayesian, CMA-ES) calls `SearchSpace.sample(trial)` to obtain a parameter dict for a given trial.

#### Parameter types (`parameters.py`)

`Parameter` is an abstract base class. Each subclass delegates to the corresponding Optuna suggestion method and supports JSON round-trips via `to_dict()` / `Parameter.from_dict()`.


| Class | Optuna method | Constructor arguments |
|---|---|---|
| `IntParameter` | `suggest_int` | `name`, `low`, `high`, `step=1`, `log=False` |
| `FloatParameter` | `suggest_float` | `name`, `low`, `high`, `log=False` |
| `LogUniformParameter` | `suggest_float(log=True)` | `name`, `low`, `high` — convenience subclass of `FloatParameter` |
| `DiscreteUniformParameter` | `suggest_float(step=q)` | `name`, `low`, `high`, `q` |
| `CategoricalParameter` | `suggest_categorical` | `name`, `choices: list[Any]` |


`Parameter.from_dict` dispatches on a `"type"` key in the serialized dict. Passing a dict without this key, or with an unrecognised value, raises a descriptive `ValueError`.

#### `SearchSpace` (`space.py`)

A named container of parameters with fluent construction, conditional sampling, and JSON persistence.

```python
space = SearchSpace(name="resnet_search")
space.add(IntParameter("num_layers", low=8, high=50))
space.add(LogUniformParameter("learning_rate", low=1e-5, high=1e-1))
space.add(CategoricalParameter("optimizer", choices=["adam", "sgd"]))
space.add(IntParameter("batch_size", low=16, high=256, step=16))

# Conditional parameter — only sampled when the predicate is True
space.add_condition(
    lambda sampled: sampled["optimizer"] == "sgd",
    FloatParameter("momentum", low=0.8, high=0.99),
)

params = space.sample(trial)   # dict[str, Any] — all unconditional params +
                                # any conditional params whose predicate fired
```


| Method | Returns | Notes |
|---|---|---|
| `add(param)` | `SearchSpace` | Registers an unconditional parameter; fluent |
| `add_condition(pred, param)` | `SearchSpace` | Registers a conditional parameter; fluent |
| `sample(trial)` | `dict[str, Any]` | Evaluates unconditional params then conditionals in registration order |
| `to_dict()` | `dict` | Serializes name, description, and unconditional params; conditions are excluded |
| `from_dict(d)` | `SearchSpace` | Reconstructs from a `to_dict()` payload |
| `save(path)` | `None` | Writes `to_dict()` as indented JSON |
| `load(path)` | `SearchSpace` | Reads and reconstructs from a JSON file |


Conditions (callable closures) are intentionally excluded from serialization — `save()`/`load()` round-trips unconditional parameters only.

#### `SearchSpaceComposer` (`composer.py`)

Static utilities for combining and projecting search spaces.


| Method | Signature | Behaviour |
|---|---|---|
| `merge` | `merge(*spaces, name)` | Union of all parameter dicts; later spaces override on name conflict; conditions from all spaces concatenated in argument order |
| `inherit` | `inherit(parent, child)` | Starts from parent's parameters, overlays child's (child wins on conflict); name and description taken from child; parent conditions registered before child conditions |
| `restrict` | `restrict(space, allowed_params)` | Returns a new space containing only the named parameters; conditions whose associated parameter is not in `allowed_params` are dropped |


```python
# Build a specialised child from a shared base
base = SearchSpace(name="base")
base.add(IntParameter("num_layers", low=4, high=32))
base.add(LogUniformParameter("learning_rate", low=1e-5, high=1e-1))

child = SearchSpace(name="deep_variant")
child.add(IntParameter("num_layers", low=64, high=512))  # overrides base

deep = SearchSpaceComposer.inherit(base, child)
# → num_layers from child (64–512), learning_rate from base
```

### Search Strategies (`search/`)

All search algorithms implement the `SearchStrategy` abstract base class. The module exports `SearchStrategy`, `RandomSearch`, `GridSearch`, `BayesianSearch`, `EvolutionarySearch`, and `MetaInformedSearch` from `orcalab.search`.

#### `SearchStrategy` (`base.py`)

Defines the four-member contract every algorithm must honour.

| Member | Kind | Description |
|---|---|---|
| `suggest(search_space)` | abstract method | Sample the next candidate `dict[str, Any]` from the space |
| `update(params, result)` | abstract method | Record the observed metric for a previously suggested candidate |
| `get_best(n=1)` | abstract method | Return the top-*n* `(params, value)` pairs, sorted by result descending |
| `n_trials` | abstract property | Number of completed (updated) trials |
| `get_history()` | concrete method | Returns `get_best(n_trials)`, or `[]` when no trials have been recorded yet |

All subclasses get `get_history()` for free; only the four abstract members need implementation.

#### `RandomSearch` (`random_search.py`)

Uniform random search backed by an internal Optuna study with `direction="maximize"`.

```python
searcher = RandomSearch(random_state=42)   # seed controls reproducibility
params   = searcher.suggest(space)         # -> dict[str, Any]
searcher.update(params, result=0.93)       # must be called in the same order as suggest()
best     = searcher.get_best(3)            # -> [(params, value), ...]  top-3 descending
```

**Pending-trial bookkeeping** — `suggest()` enqueues the Optuna trial alongside the returned params dict on an internal `deque`; `update()` pops from the front (FIFO). `update()` validates that the supplied `params` match the head of the queue and raises `ValueError` on a mismatch, guarding against out-of-order calls. Calling `update()` with no prior `suggest()` also raises `ValueError`.

`get_best(n)` sorts all `TrialState.COMPLETE` study trials by value descending and returns the top-*n*; if fewer than *n* completed trials exist it returns all of them.

Constructing `RandomSearch` sets Optuna's **global** logging level to `WARNING`, suppressing per-trial INFO output across the process.

#### `GridSearch` (`grid_search.py`)

Exhaustive search over the full Cartesian product of discretized parameter values. The grid is built lazily on the first `suggest()` call using the public `SearchSpace.to_dict()` / `Parameter.from_dict()` API — no private attribute access.

```python
searcher = GridSearch(n_steps=5)   # n_steps controls continuous-parameter resolution
try:
    while True:
        params = searcher.suggest(space)
        searcher.update(params, result=train(params))
except StopIteration:
    best = searcher.get_best(1)
```

**Discretization rules:**

| Parameter type | Grid values |
|---|---|
| `CategoricalParameter` | All `choices` values, in declaration order |
| `DiscreteUniformParameter` | `[round(low + i·q, 10)  for i in range(round((high−low)/q) + 1)]` |
| `IntParameter` with `step > 1` | `list(range(low, high+1, step))` |
| `IntParameter` with `step == 1` | `n_steps` evenly-spaced integers (linspace); full range used when range < `n_steps` |
| `FloatParameter` / `LogUniformParameter` | `n_steps` linspace values; log-spaced when `param.log is True` |

`suggest()` returns grid entries in Cartesian-product order; `StopIteration` is raised once the grid is exhausted. `_grid_values()` raises `TypeError` for any parameter type not covered by the table above, failing fast rather than silently falling back to incorrect defaults.

#### `BayesianSearch` (`bayesian.py`)

Bayesian optimisation backed by Optuna's Tree-structured Parzen Estimator (TPE) sampler. TPE models the distribution of good and bad hyperparameter configurations separately and uses that model to propose the next candidate, making it substantially more sample-efficient than random search after a short warm-up phase. All Optuna storage backends are supported, enabling study persistence and cross-process resume.

```python
searcher = BayesianSearch(
    study_name="my_sweep",              # identifies the study in the Optuna backend
    direction="maximize",               # or "minimize"
    storage="sqlite:///sweep.db",       # optional — omit for in-memory
    warm_start_trials=[                 # optional prior (params, value) pairs
        ({"lr": 0.01, "layers": 4}, 0.91),
    ],
)
params = searcher.suggest(space)        # -> dict[str, Any]
searcher.update(params, result=0.93)
best   = searcher.get_best(3)           # top-3 by value, direction-aware

# Inject priors independently (e.g. from OrcaMind warm-start)
searcher.inject_priors(prior_list, search_space=space)

# Access the underlying Optuna study for advanced inspection
print(searcher.study.best_trial)
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `study_name` | `str` | `"orcalab_bayesian"` | Identifies the study within the Optuna backend; reusing the same name and `storage` resumes an existing study |
| `direction` | `str` | `"maximize"` | Optimisation direction — `"maximize"` for accuracy-style metrics, `"minimize"` for loss-style metrics |
| `sampler` | `BaseSampler \| None` | `None` → `TPESampler()` | Optuna sampler; any `BaseSampler` subclass is accepted (e.g. `CmaEsSampler`, `NSGAIISampler`) |
| `storage` | `str \| None` | `None` → in-memory | Any Optuna storage URL (e.g. `sqlite:///sweep.db`, `postgresql+psycopg2://...`) |
| `warm_start_trials` | `list[tuple[dict, float]] \| None` | `None` | Historical `(params, value)` pairs stored as deferred priors and injected as completed `FrozenTrial`s at the first `suggest()` call, before any new Optuna trial is asked |

**Methods and properties:**

| Member | Kind | Description |
|---|---|---|
| `suggest(search_space)` | method | Calls `study.ask()`, samples via `search_space.sample(trial)`, enqueues the `(params, trial)` pair on an internal FIFO deque, and returns the params dict. Raises `ValueError` if the parameter schema differs from the first call. |
| `update(params, result)` | method | Pops the oldest pending trial, validates that `params` matches it (FIFO contract), then calls `study.tell()`. NaN or ±Inf results are reported to Optuna as `TrialState.FAIL` rather than crashing; those trials do not contribute to `n_trials` or `get_best()`. Raises `ValueError` on param mismatch or when called with no pending trial. |
| `get_best(n=1)` | method | Filters `TrialState.COMPLETE` trials and returns the top-*n* `(params, value)` tuples sorted in the study's optimisation direction (descending for `maximize`, ascending for `minimize`). Returns all completed trials when fewer than *n* exist. Raises `ValueError` when `n < 1`. |
| `inject_priors(warm_trials, search_space=None)` | method | Seeds the Optuna study with historical `(params, value)` observations by constructing `FrozenTrial` objects via `optuna.trial.create_trial()` and calling `study.add_trial()`. The `search_space` argument is required when called before the first `suggest()`; afterwards the internally stored space is used. Raises `ValueError` for any non-finite value. |
| `n_trials` | property | Count of `TrialState.COMPLETE` trials. FAIL trials (from NaN/Inf results) are excluded. |
| `study` | property | Exposes the underlying `optuna.Study` directly — useful for inspecting `study.best_trial`, plotting with Optuna's visualisation module, or passing to external tooling. |

**`_build_distributions(space)` helper** — module-level function that converts a `SearchSpace`'s parameter objects into the `optuna.distributions` types required by `optuna.trial.create_trial()` when constructing `FrozenTrial`s for `inject_priors()`.

| Parameter type | Optuna distribution |
|---|---|
| `CategoricalParameter` | `CategoricalDistribution(choices)` |
| `DiscreteUniformParameter` | `FloatDistribution(low, high, step=q)` |
| `FloatParameter` / `LogUniformParameter` | `FloatDistribution(low, high, log=log)` |
| `IntParameter` | `IntDistribution(low, high, step=step, log=log)` |

**Persistence** — when `storage` is provided the Optuna study survives process restarts. Constructing a new `BayesianSearch` with the same `study_name` and `storage` loads the existing study via `load_if_exists=True`; all previously completed trials are immediately available through `n_trials` and `get_best()`.

**Guardrails added by CodeRabbit review:**

- *Schema stability in `suggest()`* — on every call after the first, the set of parameter names derived from `_build_distributions(search_space)` is compared against the set from the stored space. A schema change raises `ValueError("SearchSpace schema changed…")`, preventing mixed-distribution corruption of the TPE surrogate model. Space name changes alone (same parameter set, different `SearchSpace.name`) are permitted.
- *NaN / ±Inf in `update()`* — non-finite results are told to Optuna as `TrialState.FAIL`; the trial is excluded from counts and rankings. Subsequent valid results record normally.
- *Non-finite values in `inject_priors()`* — raises `ValueError("Warm-start value must be finite…")` immediately, enforcing consistency with the live `update()` contract and preventing polluted rankings.
- *`get_best(n)` input validation* — raises `ValueError("n must be >= 1.")` for `n < 1`, preventing silent empty-slice semantics.

#### `EvolutionarySearch` (`evolutionary.py`)

Evolution-strategy-based optimisation backed by CMA-ES (Covariance Matrix Adaptation Evolution Strategy) via the `cma` library. CMA-ES maintains a multivariate Gaussian distribution over the normalised parameter space and iteratively updates its mean and full covariance matrix toward regions of high fitness. It is particularly effective for moderate-dimensional continuous and mixed spaces with correlated parameters.

Because CMA-ES operates on a continuous, unconstrained Euclidean space, `EvolutionarySearch` encodes the heterogeneous `SearchSpace` into a normalised `[0, 1]^d` vector and decodes CMA-ES solutions back to parameter dicts after each generation.

```python
from orcalab.search import EvolutionarySearch

searcher = EvolutionarySearch(
    population_size=10,   # individuals per CMA-ES generation
    sigma0=0.3,           # initial step size in normalised space
    direction="maximize", # or "minimize"
    seed=42,
)
params = searcher.suggest(space)        # -> dict[str, Any]
searcher.update(params, result=0.93)
best   = searcher.get_best(3)           # top-3 by value, direction-aware
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `population_size` | `int` | `10` | Number of candidate solutions evaluated per CMA-ES generation. Must be `> 0`. |
| `sigma0` | `float` | `0.3` | Initial standard deviation (step size) of the Gaussian in the normalised `[0, 1]^d` space. Must be `> 0`. |
| `seed` | `int` | `42` | Master seed for the CMA-ES solver and the NumPy RNG used during convergence restarts. |
| `direction` | `str` | `"maximize"` | Optimisation direction — `"maximize"` for accuracy-style metrics, `"minimize"` for loss-style metrics. |

**Encoding scheme** — `_build_dim_map()` walks the `SearchSpace` parameters in declaration order and allocates dimensions in the normalised vector:

| Parameter type | Vector dimensions | Encoding / decoding |
|---|---|---|
| `CategoricalParameter` (N choices) | N | One-hot: the active choice's slot is `1.0`, all others `0.0`; decoded via `argmax` |
| `IntParameter` (linear) | 1 | `(v − low) / (high − low)`; decoded with `clip → linear → round → clip` |
| `IntParameter` (log-scale) | 1 | `(log v − log low) / (log high − log low)`; decoded with `clip → exp → round → clip` |
| `FloatParameter` (linear) | 1 | `(v − low) / (high − low)`; decoded with `clip → linear → clip` |
| `FloatParameter` (log-scale) | 1 | `(log v − log low) / (log high − log low)`; decoded with `clip → exp → clip` |
| `DiscreteUniformParameter` | 1 | Linear, treated identically to a non-log `FloatParameter` |

**Methods and properties:**

| Member | Kind | Description |
|---|---|---|
| `suggest(search_space)` | method | Returns the next parameter dict from the internal solution queue. On the first call, builds the dimension map, initialises the CMA-ES solver with `x0 = [0.5]^d`, and pre-fills the queue via `es.ask()`. Repopulates via `es.ask()` whenever the queue is empty. Raises `ValueError` if a different `SearchSpace` instance is passed after the first call, or if `_stopped` is `True` while pending trials still exist. |
| `update(params, result)` | method | Records the result for the oldest pending trial (FIFO). NaN and ±Inf results are silently dropped and do not contribute to `n_trials`, history, or the CMA-ES update. Once `population_size` valid results accumulate, calls `es.tell()` with direction-adjusted fitnesses (negated for `"maximize"`, since CMA-ES minimises internally). After each `tell()`, inspects `es.stop()` and sets `_stopped = True` if convergence is detected. Raises `ValueError` on param mismatch or when called with no pending trial. |
| `get_best(n=1)` | method | Sorts all recorded history entries by result in the optimisation direction and returns the top-*n* `(params, value)` tuples. Returns all recorded trials when fewer than *n* exist. |
| `n_trials` | property | Number of valid (non-NaN/Inf) trials recorded in history. |

**Population lifecycle:**

```
suggest() × population_size:
  ├── es.ask() → fills _solution_queue with (vec, params) pairs
  └── pops one pair per call → appends (params, vec) to _pending (FIFO)

update() × population_size valid results:
  ├── validates params against _pending head, pops
  ├── accumulates (vec, fitness) in _gen_accumulator
  └── when full: es.tell(vecs, direction-adjusted fitnesses)
                 es.stop() → sets _stopped if converged
```

**Convergence and restart** — after `es.tell()`, `es.stop()` is inspected each generation. A non-empty stop dict (conditions such as `tolfun`, `tolx`, `maxiter`) sets `_stopped = True`. On the next `suggest()` call once all pending trials are drained, a new CMA-ES instance is seeded from a Gaussian-perturbed encoding of the best-known parameters (`best_vec + N(0, sigma0)`), clipped to `[0, 1]^d`. If no history exists, a uniform random starting point is used. The `_solution_queue` is cleared before repopulating to prevent stale candidates from the old solver from entering the new generation.

**Guardrails added by CodeRabbit review:**

- *Input validation in `__init__()`* — `population_size ≤ 0` or `sigma0 ≤ 0` raises `ValueError` immediately, preventing silent failures deep inside the CMA-ES solver.
- *Single-space contract in `suggest()`* — stores the first `SearchSpace` instance; raises `ValueError` if a different instance is passed on a later call, preventing mixed-encoding corruption of the covariance model.
- *Restart guard and stale-queue eviction in `suggest()`* — if `_stopped` is `True` and `_pending` is non-empty, `suggest()` raises `ValueError` rather than mixing vectors from the old and new solvers. When restart proceeds, `_solution_queue.clear()` is called before `es.ask()` so stale generation-N candidates are never served from the new solver.

---

### Pruning Strategies (`pruning/`)

Early-stopping strategies terminate underperforming trials before they exhaust their resource budget, recovering compute that would otherwise be wasted running trials that cannot plausibly converge to a competitive result. All three concrete strategies implement the `Pruner` ABC; the module re-exports all four public names from `orcalab.pruning`.

#### `Pruner` (`base.py`)

Abstract base class defining the shared contract for all pruning strategies.

| Member | Kind | Description |
|---|---|---|
| `should_prune(trial_id, step, current_value, all_trial_values)` | abstract method | Return `True` if the trial should be stopped at this step. `all_trial_values` is `dict[str, list[float]]` — the full observed history of every active trial indexed by trial ID. |
| `name` | abstract property | Strategy identifier string. |

Passing the full `all_trial_values` dict into every call allows each strategy to compare relative progress across the live cohort without shared mutable state between trials.

#### `MedianStoppingPruner` (`median.py`)

Prunes a trial when its current value falls **strictly below** the median of every peer's best observed value up to the current step. Peers with shorter histories contribute their maximum available value, so mid-run trials are never artificially excluded from the comparison pool.

```python
from orcalab.pruning import MedianStoppingPruner

pruner = MedianStoppingPruner(warmup_steps=5)
should_stop = pruner.should_prune(
    trial_id="trial_42",
    step=10,
    current_value=0.71,
    all_trial_values={
        "trial_0": [0.82, 0.85, 0.87, 0.88, 0.90, 0.91, 0.91, 0.92, 0.92, 0.93],
        "trial_1": [0.78, 0.80, 0.83, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91],
    },
)
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `warmup_steps` | `int` | `5` | No pruning decision is issued for steps `< warmup_steps`. Raises `ValueError` if negative. |

**Behaviour:**

- Steps strictly less than `warmup_steps` always return `False`.
- If no qualifying peers exist (empty `all_trial_values` or every peer has an empty history) returns `False`.
- The current trial is excluded from the peer set — its own history does not influence the median.
- Comparison is `current_value < median`; a value equal to the median is **not** pruned.
- Each peer contributes `max(values[:step])` — its best result within the observable window, not necessarily the value at exactly step `s`.

#### `ASHAPruner` (`asha.py`)

Asynchronous Successive Halving Algorithm (Li et al., 2018). Evaluates each trial only at **rung levels** (`min_resource × reduction_factor^k` for k = 0, 1, 2, …); all other steps are passed through at zero cost. At each rung the top `1/reduction_factor` fraction of competing trials is promoted and the rest are pruned.

```python
from orcalab.pruning import ASHAPruner

pruner = ASHAPruner(min_resource=1, max_resource=81, reduction_factor=3)
# Rung schedule: steps 1, 3, 9, 27, 81
# At each rung: top 1/3 promoted, bottom 2/3 pruned

should_stop = pruner.should_prune(
    trial_id="trial_0",
    step=1,
    current_value=0.55,
    all_trial_values={"trial_1": [0.72], "trial_2": [0.81]},
)
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `min_resource` | `int` | `1` | Step number of the first rung. Raises `ValueError` if < 1. |
| `max_resource` | `int` | `81` | Rungs are only generated up to and including this step. Raises `ValueError` if < `min_resource`. |
| `reduction_factor` | `int` | `3` | Inverse of the promoted fraction at each rung (`1/reduction_factor` kept). Raises `ValueError` if < 2. |

**Rung schedule (default params):**

| Rung | Step | Trials kept (of N reaching this rung) |
|---|---|---|
| 0 | 1 | `max(1, N // 3)` |
| 1 | 3 | `max(1, N // 3)` |
| 2 | 9 | `max(1, N // 3)` |
| 3 | 27 | `max(1, N // 3)` |
| 4 | 81 | `max(1, N // 3)` |

**Internal state:** `_promoted: dict[int, list[str]]` records which trial IDs have been promoted through each rung, making promotion history inspectable without re-evaluating previously decided trials.

**Key invariants:**
- A non-rung step always returns `False` immediately — zero overhead between rungs.
- `keep = max(1, n // reduction_factor)` — at least one trial always survives, even when only a single trial has reached the rung.
- Steps beyond `max_resource` are not rung levels and always return `False`.

**Compute savings** — in a 20-trial sweep with sequential best-first execution, ASHA executes ~100 total steps vs. 1,620 for unpruned runs (>93% savings). The `TestASHAPruningSavings` class in `tests/performance/test_pruning_savings.py` drives a deterministic concave-quadratic synthetic sweep and makes four executable assertions: (1) ≥40% compute savings for 20 trials — a conservative threshold that holds even under concurrent-execution orderings where the best trial is not always evaluated first; (2) the highest-quality trial is never pruned and always runs to `max_resource`; (3) the lowest-quality trial is pruned before completion once a strong competitor has run; (4) savings with 27 trials meet the ≥40% threshold and are at least as large as with 20 trials, enforcing the monotonicity property directly.

#### `MetaPruner` (`meta_pruner.py`)

Wraps any `Pruner` with an OrcaMind performance-prediction layer. Before delegating to the base pruner, `MetaPruner` queries `OrcaMindClient.predict_performance` using the trial's observed value history as the task embedding. If the predicted final performance is below `prediction_threshold`, the trial is pruned immediately — potentially several rungs earlier than a rung-based strategy alone would trigger.

```python
from orca_shared.clients.orcamind_client import OrcaMindClient
from orcalab.pruning import ASHAPruner, MetaPruner

client = OrcaMindClient(base_url="http://localhost:8000")
base   = ASHAPruner(min_resource=1, max_resource=81, reduction_factor=3)
pruner = MetaPruner(
    orcamind_client=client,
    base_pruner=base,
    prediction_threshold=0.3,
    min_steps_before_prediction=10,
)
should_stop = pruner.should_prune(
    trial_id="trial_0",
    step=15,
    current_value=0.41,
    all_trial_values={"trial_0": [0.31, 0.35, 0.38, ...]},
)
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `orcamind_client` | `OrcaMindClient` | required | Async HTTP client — queries `/api/v1/predict-performance`. |
| `base_pruner` | `Pruner` | required | Fallback strategy when OrcaMind is unavailable or the prediction is above threshold. |
| `prediction_threshold` | `float` | `0.3` | Predicted final performance strictly below this value triggers immediate pruning. |
| `min_steps_before_prediction` | `int` | `10` | Steps below this are never sent to OrcaMind; too-short curves carry insufficient signal. |

**Decision flow:**

```text
should_prune(trial_id, step, current_value, all_trial_values)
  │
  ├─ step < min_steps_before_prediction  →  return False   (warmup — no query)
  │
  ├─ _query_orcamind(...)
  │    ├─ task_embedding = observed_curve + [current_value]
  │    ├─ asyncio.new_event_loop().run_until_complete(predict_performance(...))
  │    └─ any Exception  →  log WARNING, return None
  │
  ├─ predicted < prediction_threshold   →  return True    (OrcaMind early stop)
  │
  └─ else  →  return base_pruner.should_prune(...)         (delegate to base)
```

**Graceful degradation** — any exception from OrcaMind (network error, timeout, malformed response) is caught inside `_query_orcamind`, logged at `WARNING`, and yields `None`. The decision then falls through to `base_pruner.should_prune()`. The sweep is never blocked by an unavailable prediction service.

**Async/sync bridge** — `OrcaMindClient.predict_performance` is `async def` but `should_prune` must be synchronous. `MetaPruner` bridges this by creating a fresh event loop per call via `asyncio.new_event_loop()`, running the coroutine inside it, and closing the loop in a `finally` block. This is safe from synchronous callers (CLI, test suite) where no event loop is already running.

#### Public API (`__init__.py`)

All four names are importable directly from `orcalab.pruning`:

```python
from orcalab.pruning import Pruner, MedianStoppingPruner, ASHAPruner, MetaPruner
```

---

### Prefect Orchestration (`orchestration/`)

The orchestration layer composes the runner, search strategies, pruners, storage backends, and OrcaMind into schedulable Prefect 2.x flows. Each flow is a self-contained unit of work that can be deployed to a Prefect work pool and triggered on demand or on a schedule; each task is a fine-grained, retriable step inside those flows.

```python
from orcalab.orchestration.tasks import prepare_data, train_model, evaluate, log_results, get_orcamind_priors
from orcalab.orchestration.flows import (
    run_single_experiment,
    run_sweep,
    meta_informed_sweep,
    continuous_learning_loop,
)
```

#### Tasks (`orchestration/tasks/`)

| Task | Decorator attributes | Signature (simplified) | Behaviour |
|---|---|---|---|
| `prepare_data` | `retries=2`, `retry_delay_seconds=30` | `(task_id: str, storage: StorageBackend) -> pd.DataFrame` | Downloads `datasets/{task_id}/data.parquet` from the storage backend and returns it as a DataFrame. Retried up to twice on transient failures. |
| `train_model` | `timeout_seconds=3600` | `(experiment: Experiment, pruner: Pruner \| None, runner: ExperimentRunner) -> ExperimentResult` | Delegates to `runner.run(experiment, pruner=pruner)`. The `pruner` parameter is wired through from the enclosing flow so ASHA/median/meta pruning applies inside the runner's epoch loop. |
| `evaluate` | — | `(result: ExperimentResult, metrics: list[str] \| None = None) -> dict[str, float \| None]` | Extracts the requested metrics from `result.metrics`. Defaults to `["accuracy", "loss"]` when `metrics` is `None`. Returns `None` for any metric not present in the result. |
| `log_results` | — | `(result: ExperimentResult, orcamind_client: OrcaMindClient) -> None` | Submits a `FeedbackRequest` to OrcaMind using `max(result.metrics.values())` as the scalar signal under the key `"objective"`. Silently swallows `httpx.ConnectError`, `httpx.TimeoutException`, and `httpx.HTTPStatusError` so transient OrcaMind failures never block flows. |
| `get_orcamind_priors` | `retries=1` | `(task_id: str, orcamind_url: str) -> list[ModelRecommendation] \| None` | Embeds the task via `embed_task` then requests a model recommendation via `recommend_model`. Returns `[ModelRecommendation]` on success or `None` on any network or HTTP error, so sweeps always start even when OrcaMind is unreachable. |

**Task notes:**

- `train_model` does not accept a `data` parameter — data is loaded once by `prepare_data` and passed directly to the `Experiment` constructor in the enclosing flow. The runner's `run()` method only takes `experiment` and `pruner`.
- `log_results` uses `max(metrics.values())` when `result.metrics` is non-empty; falls back to `0.0` for an empty metrics dict.
- `get_orcamind_priors` returns a single-element list (not a bare `ModelRecommendation`) so callers can uniformly check for `None` (failure) vs. a non-empty list (success). The `retries=1` decorator means the task will attempt the OrcaMind calls twice before returning `None`.

#### Integration Tests ( ↔ OrcaMind)

 contains 20 tests that validate the complete bidirectional OrcaLab ↔ OrcaMind contract using  to intercept all httpx calls at the network layer — no running OrcaMind service is required:

| Test class | Tests | Covers |
|---|---|---|
|  | 3 | Priors injected into base strategy; all three OrcaMind endpoints called;  works after warm-start |
|  | 3 | 5xx on recommend-model, 503 during active sweep,  on embed — all fall back to zero priors |
|  | 3 | One feedback request per completed trial; correct payload shape (, , ); no requests when zero trials |
|  | 6 | Happy-path returns ; both embed and recommend called; , ,  each return ;  set |
|  | 5 |  called once; max metric used as feedback value; , ,  each swallowed |

A Prefect stub  in  installs a lightweight fake  module into  before any orchestration import, supporting both  factory and bare  decorator styles.

---

#### Flows (`orchestration/flows/`)

**`run_single_experiment` (`single_experiment.py`)**

End-to-end single-trial flow.

```python
@flow(name="single_experiment")
async def run_single_experiment(
    task_id: str,
    model_config: dict,
    training_config: dict,
    *,
    storage: StorageBackend | None = None,
    pruner: Pruner | None = None,
    runner: ExperimentRunner,
    orcamind_client: OrcaMindClient | None = None,
) -> ExperimentResult:
```

| Parameter | Default | Notes |
|---|---|---|
| `storage` | `None` | When `None`, skips `prepare_data` and passes an empty `DataFrame` to the experiment. |
| `pruner` | `None` | When `None`, no early stopping is applied; wired through to `train_model`. |
| `orcamind_client` | `None` | When `None`, skips `log_results`. |

Execution steps:
1. If `storage` is provided: call `prepare_data(task_id, storage)` to fetch the dataset.
2. Parse `task_id` as a UUID for `Experiment.task_id`; fall back to `None` if the format is invalid.
3. Construct an `Experiment` from `model_config` and `training_config`.
4. Call `train_model(experiment, pruner, runner)` → `ExperimentResult`.
5. Call `evaluate(result)`.
6. If `orcamind_client` is provided: call `log_results(result, orcamind_client)`.
7. Return the `ExperimentResult`.

---

**`run_sweep` (`sweep.py`)**

N-trial hyperparameter sweep with configurable search strategy and pruner.

```python
@flow(name="hyperparameter_sweep")
async def run_sweep(
    task_id: str,
    search_space: SearchSpace,
    n_trials: int = 50,
    strategy: str = "bayesian",
    pruner_name: str = "asha",
    *,
    storage: StorageBackend | None = None,
    runner: ExperimentRunner,
    orcamind_client: OrcaMindClient | None = None,
) -> list[ExperimentResult]:
```

Strategy and pruner selection is handled by two module-level factory helpers:

**`_build_strategy(name)`** — dispatches on the `strategy` string:

| `strategy` value | Object returned |
|---|---|
| `"bayesian"` (default) | `BayesianSearch()` |
| `"random"` | `RandomSearch()` |
| `"grid"` | `GridSearch()` |
| `"evolutionary"` | `EvolutionarySearch()` |
| any other value | `BayesianSearch()` (fallback) |

**`_build_pruner(name, orcamind_client)`** — dispatches on the `pruner_name` string:

| `pruner_name` value | `orcamind_client` | Object returned |
|---|---|---|
| `"asha"` | any | `ASHAPruner()` |
| `"median"` | any | `MedianStoppingPruner()` |
| `"meta"` | provided | `MetaPruner(orcamind_client=client, base_pruner=ASHAPruner())` |
| `"meta"` | `None` | `ASHAPruner()` (fallback — no client available) |
| anything else | any | `None` (no pruning) |

Execution: data is loaded once via `prepare_data` (or empty DataFrame if `storage=None`) and reused across all `n_trials`. Per trial: `strategy.suggest(search_space)` → construct `Experiment` → `train_model` (with pruner) → `evaluate` → `log_results` (if client present) → `strategy.update(params, result)`.

---

**`meta_informed_sweep` (`meta_sweep.py`)**

OrcaMind-warm-started sweep that initialises the search strategy from prior experiment results.

```python
@flow(name="meta_informed_sweep")
async def meta_informed_sweep(
    task_id: str,
    n_trials: int = 50,
    use_orcamind: bool = True,
    *,
    search_space: SearchSpace | None = None,
    runner: ExperimentRunner,
    orcamind_client: OrcaMindClient | None = None,
) -> list[ExperimentResult]:
```

| Path | Condition | Strategy | Pruner | OrcaMind calls |
|---|---|---|---|---|
| OrcaMind-enabled | `use_orcamind=True` and `orcamind_client` provided | `MetaInformedSearch` warm-started via `initialize_from_orcamind(task_id, search_space)` | `MetaPruner` | `initialize_from_orcamind` before loop; `flush_results_to_orcamind(task_id)` after loop |
| Fallback | `use_orcamind=False` or no client | `BayesianSearch` | `ASHAPruner` | None |

Returns the top-5 results sorted by accuracy descending.

---

**`continuous_learning_loop` (`continuous_learning.py`)**

Outer scheduling loop that calls `meta_informed_sweep` for every task in every iteration.

```python
@flow(name="continuous_learning")
async def continuous_learning_loop(
    task_ids: list[str],
    iterations: int = 10,
    trials_per_iteration: int = 20,
    iteration_sleep_seconds: float = 60.0,
    *,
    runner: ExperimentRunner,
    orcamind_client: OrcaMindClient | None = None,
) -> None:
```

For each of `iterations` iterations, `meta_informed_sweep` is called for every `task_id` in `task_ids`. The flow sleeps `iteration_sleep_seconds` between iterations but **not** after the final iteration. Total sweep calls = `iterations × len(task_ids)`.

#### Deployment Configuration (`prefect.yaml`)

All four flows are declared as named deployments on the `orcalab-pool` work pool.

| Deployment name | Entry point |
|---|---|
| `single-experiment` | `single_experiment.py:run_single_experiment` |
| `hyperparameter-sweep` | `sweep.py:run_sweep` |
| `meta-informed-sweep` | `meta_sweep.py:meta_informed_sweep` |
| `continuous-learning` | `continuous_learning.py:continuous_learning_loop` |

Apply all deployments with:

```bash
prefect deploy --all
```

---

### Streamlit Dashboard (`orcalab.visualization`)

A four-page Streamlit application that provides live experiment observability for an OrcaLab deployment. Every page is structured as a collection of named, pure data-processing functions (fetch helpers and transform helpers) plus a single `_page()` render function guarded by `if __name__ == "__main__": _page()`. This pattern lets the unit test suite import and call the data functions directly, without requiring a live Streamlit runtime or Plotly installation.

The app is launched via the `orcalab dashboard` CLI command or directly:

```bash
streamlit run orcalab/visualization/app.py  # binds to port 8502 by default
```

#### Application Entry Point (`app.py`)

```python
st.set_page_config(page_title="OrcaLab", layout="wide")
st.sidebar.text_input("OrcaLab API URL", value="http://localhost:8001")
pg = st.navigation([
    st.Page("pages/live_experiments.py", title="Live Experiments"),
    st.Page("pages/search_progress.py",  title="Search Progress"),
    st.Page("pages/results_explorer.py", title="Results Explorer"),
    st.Page("pages/meta_analysis.py",    title="Meta-Analysis"),
])
if __name__ == "__main__":
    pg.run()
```

The sidebar API URL is declared here and mirrored inside each page module so the current page always has the value in scope regardless of navigation order.

---

#### Reusable Chart Components (`components/`)

All three component modules expose pure functions that return a Plotly `go.Figure`. They have no Streamlit dependency and can be called from any context.

##### `metric_plots.py`

| Function | Signature | Description |
|---|---|---|
| `loss_curve` | `(history: list[dict], title: str = "Training Loss") -> go.Figure` | Multi-series line chart — one `go.Scatter` trace per metric key found across all history dicts. The epoch key is `"epoch"` when any row contains that key, falling back to `"step"`. Metric keys are the sorted union across every row (excluding the epoch key and `"run_id"`). Rows where the epoch or value is `None`/`NaN` are skipped per-series. Returns an empty, titled figure for an empty history list. |
| `metric_comparison` | `(results: list[ExperimentResult], metric: str) -> go.Figure` | Bar chart with one `go.Bar` trace. Labels are the first 8 characters of each `experiment_id` UUID. Experiments with `metrics=None`, missing the requested key, or a `NaN` value are silently excluded. Returns an empty figure when no valid results remain. |

##### `parallel_coords.py`

| Function | Signature | Description |
|---|---|---|
| `_safe_float` | `(v: object) -> float` | Module-private helper. Converts `v` to `float`, returning `math.nan` for `None` or any unconvertible value. Used internally to normalise the `objective` column without raising on malformed API payloads. |
| `parallel_coordinates` | `(trials: list[dict], colorscale: str = "Viridis") -> go.Figure` | `go.Parcoords` chart. Parameter dimensions are the sorted union of all keys across every trial excluding `"objective"`. String-valued parameters are encoded as integer codes with `ticktext` labels (sorted by string representation to handle mixed types). Numeric parameters are passed as floats with `NaN` for missing values. The `objective` column drives the line colour via a continuous colourscale with `showscale=True`. Returns an empty figure for an empty trials list. |

##### `pareto_frontier.py`

| Function | Signature | Description |
|---|---|---|
| `_is_pareto_optimal` | `(costs: list[tuple[float, float]]) -> list[bool]` | O(n²) domination check under minimisation on both axes. Point `j` dominates point `i` iff `xⱼ ≤ xᵢ` and `yⱼ ≤ yᵢ` with at least one strict inequality. Returns a `list[bool]` of the same length. |
| `pareto_plot` | `(results: list[ExperimentResult], x_metric: str, y_metric: str) -> go.Figure` | Two-trace scatter. Sub-optimal experiments are rendered as steelblue circles (size 8); Pareto-optimal experiments as red diamonds (size 10). Experiments with `metrics=None`, a missing axis metric, or any `NaN` value on either axis are excluded before the frontier calculation. Axis titles are set to `x_metric` and `y_metric`. Returns an empty figure with axis labels when no valid points exist. |

---

#### Dashboard Pages (`pages/`)

All pages use `requests.get` (synchronous) for API calls, consistent with the OrcaMind dashboard pattern.

##### `live_experiments.py` — Live Experiments

Auto-refreshing view of all running and recent experiments.

**Pure functions:**

| Function | Signature | Description |
|---|---|---|
| `fetch_experiments` | `(api_url: str) -> list[dict]` | `GET {api_url}/api/v1/experiments`. Raises on any HTTP error. |
| `fetch_experiment_history` | `(api_url: str, experiment_id: str) -> list[dict]` | `GET {api_url}/api/v1/experiments/{experiment_id}` — returns the experiment record; callers extract the `metrics` JSONB field (most recent epoch snapshot) as a single-element history list. For real-time per-epoch streaming, use `WS /api/v1/experiments/{id}/live` instead. |
| `color_for_status` | `(status: str) -> str` | Case-insensitive lookup into `STATUS_COLORS`: `RUNNING` → `#28a745`, `PENDING` → `#6c757d`, `FAILED` → `#dc3545`, `COMPLETED` → `#007bff`. Unknown statuses fall back to gray. |
| `compute_progress` | `(current_epoch: int \| None, total_epochs: int \| None) -> float` | Returns `float(current_epoch) / float(total_epochs)` clamped to `[0.0, 1.0]`. Returns `0.0` when either argument is `None` or `total_epochs ≤ 0`. Negative epoch values are clamped to `0.0`. |

**`_page()` flow:**
1. Fetch all experiments; call `st.error` + `st.stop` on failure.
2. Status-filter dropdown; `st.dataframe` of filtered results.
3. Selectbox for per-experiment detail — HTML-escaped coloured status label (XSS-safe), `st.progress` bar with `Epoch N / M` label (explicit `is None` guard preserves epoch `0`).
4. `loss_curve()` chart from per-experiment history.
5. When the auto-refresh checkbox is enabled: `time.sleep(5)` + `st.rerun()`.

---

##### `search_progress.py` — Search Progress

Hyperparameter sweep visualisation with parallel coordinates.

**Pure functions:**

| Function | Signature | Description |
|---|---|---|
| `fetch_sweeps` | `(api_url: str) -> list[dict]` | `GET {api_url}/api/v1/sweeps`. |
| `fetch_sweep_trials` | `(api_url: str, sweep_id: str) -> list[dict]` | `GET {api_url}/api/v1/sweeps/{sweep_id}/trials`. |
| `find_best_trial` | `(trials: list[dict]) -> dict \| None` | Returns the trial with the highest numeric `objective`. Each trial's objective is converted to `float` inside a `try/except`; non-numeric values (strings, `None`) are skipped. Returns `None` when no valid numeric objective exists. The returned dict always has `objective` as a `float`. |
| `build_cumulative_df` | `(trials: list[dict]) -> pd.DataFrame` | Returns `DataFrame[trial_index, cumulative_count]` with 1-based sequential indices. Returns an empty DataFrame with the same columns for an empty trials list. |

**`_page()` flow:**
1. Sweep selectbox → `fetch_sweep_trials()`.
2. `parallel_coordinates()` chart for all trials.
3. `st.sidebar.metric("Best Objective", f"{best['objective']:.4f}")` when a best trial exists.
4. `px.line` cumulative trial count chart.

---

##### `results_explorer.py` — Results Explorer

Filterable table of completed experiments with side-by-side A/B config diff.

**Pure functions:**

| Function | Signature | Description |
|---|---|---|
| `fetch_completed_experiments` | `(api_url: str) -> list[dict]` | `GET {api_url}/api/v1/experiments` with `params={"status": "COMPLETED"}`. |
| `filter_experiments` | `(experiments, *, task_id, domain, date_from, date_to) -> list[dict]` | Applies up to four optional filters. Experiments missing the `completed_at` field are excluded when either date filter is active. Unparseable `completed_at` strings are excluded rather than raising. |
| `diff_configs` | `(exp_a: dict, exp_b: dict) -> dict` | Returns `{key: {"a": val_a, "b": val_b}}` for every key where the two dicts differ, sorted alphabetically. Returns `{}` for identical inputs. |

**`_page()` flow:**
1. Four-column filter row (task_id, domain, date_from, date_to) → filtered experiment table.
2. Two selectboxes (A / B) drawn from the filtered list.
3. `diff_configs()` result as `st.json()`; `metric_comparison()` chart for a user-chosen metric.
4. Failed `ExperimentResult` construction is caught and logged at `WARNING` level rather than silently swallowed.

---

##### `meta_analysis.py` — Meta-Analysis

Cross-experiment aggregate views: heatmap, scatter, and improvement trend.

**Pure functions:**

| Function | Signature | Description |
|---|---|---|
| `fetch_all_experiments` | `(api_url: str) -> list[dict]` | `GET {api_url}/api/v1/experiments`. |
| `build_domain_arch_heatmap` | `(experiments: list[dict], metric: str = "accuracy") -> pd.DataFrame` | Pivot table — rows are domains, columns are architectures, values are mean metric. Experiments missing `domain`, `architecture`, or the requested metric are excluded. Returns an empty `DataFrame` when no valid records exist. |
| `build_scatter_df` | `(experiments: list[dict], metric: str = "accuracy") -> pd.DataFrame` | Returns `DataFrame[complexity, accuracy, experiment_id]` where `complexity = int(n_features) × int(n_samples)`. Both `n_features` and `n_samples` must be present and numeric; either missing or non-integer-convertible value excludes the record. |
| `build_trend_df` | `(experiments: list[dict], metric: str = "accuracy") -> pd.DataFrame` | Returns `DataFrame[completed_at, value, best_so_far]` sorted ascending by `completed_at`. `best_so_far` is the running cumulative maximum of `value`. `completed_at` is parsed via `pd.to_datetime(..., errors="coerce")`; NaT results exclude the record. |

**`_page()` flow:**
1. `go.Heatmap` (RdYlGn colourscale) — NaN cells are converted to `None` for Plotly compatibility.
2. `px.scatter` of task complexity vs. accuracy.
3. `px.line` of `best_so_far` over time.

---

#### Testing Infrastructure

The visualization test suite is self-contained under `tests/unit/visualization/` and requires no live Streamlit, Plotly, or OrcaLab API.

**`conftest.py`** — session-scoped autouse `_patch_streamlit` fixture:

```python
_MOCKED_MODULES = (
    "streamlit", "streamlit.components", "streamlit.components.v1",
    "streamlit.testing", "streamlit.testing.v1",
    "plotly", "plotly.express", "plotly.graph_objects",
)

@pytest.fixture(scope="session", autouse=True)
def _patch_streamlit():
    originals = {mod: sys.modules.get(mod) for mod in _MOCKED_MODULES}
    for mod in _MOCKED_MODULES:
        sys.modules[mod] = MagicMock()
    sys.modules["streamlit"] = mock_st  # returned for per-test assertions
    yield mock_st
    for mod, original in originals.items():
        if original is None:
            sys.modules.pop(mod, None)
        else:
            sys.modules[mod] = original   # restore pre-existing entries
```

Each test file uses a **module-scoped fixture** that pops its target module from `sys.modules` and re-imports it cleanly, preventing stale mock state from leaking between test files:

```python
@pytest.fixture(scope="module")
def mp(_patch_streamlit):
    sys.modules.pop("orcalab.visualization.components.metric_plots", None)
    return importlib.import_module("orcalab.visualization.components.metric_plots")
```

API calls inside page modules are patched at call-site using `unittest.mock.patch("requests.get", return_value=mock_resp)` so network calls never reach a real server.

**Test coverage summary:**

| File | Tests | Covers |
|---|---|---|
| `components/test_metric_plots.py` | 13 | `loss_curve` (7), `metric_comparison` (6) |
| `components/test_parallel_coords.py` | 6 | `parallel_coordinates` — empty, numeric, categorical, missing objective, colorscale forwarding, default |
| `components/test_pareto_frontier.py` | 11 | `_is_pareto_optimal` (5), `pareto_plot` (6) |
| `pages/test_app.py` | 6 | Import, `set_page_config`, title, navigation call, 4-page count, sidebar URL |
| `pages/test_live_experiments.py` | 16 | `fetch_experiments` (3), `fetch_experiment_history` (3), `color_for_status` (6), `compute_progress` (7, including negative epoch) |
| `pages/test_search_progress.py` | 15 | `fetch_sweeps` (3), `fetch_sweep_trials` (3), `find_best_trial` (5, including non-numeric objectives), `build_cumulative_df` (5) |
| `pages/test_results_explorer.py` | 19 | `fetch_completed_experiments` (4), `filter_experiments` (10), `diff_configs` (5) |
| `pages/test_meta_analysis.py` | 23 | `fetch_all_experiments` (3), `build_domain_arch_heatmap` (8), `build_scatter_df` (6, including partial complexity), `build_trend_df` (8, including invalid timestamps) |
| **Total** | **115** | |

---

### REST API (`orcalab.api`)

10 REST endpoints and 1 WebSocket endpoint served by FastAPI, documented at `GET /docs`. The service runs on port 8001 by default and is launched via `orcalab serve` or `uvicorn orcalab.api.main:app`.


| Method   | Path                                 | Status | Description                                                                                                   |
|----------|--------------------------------------|--------|---------------------------------------------------------------------------------------------------------------|
| `GET`    | `/`                                  | 200    | Service info (name, version, status)                                                                          |
| `GET`    | `/health`                            | 200    | Liveness probe — checks DB via `SELECT 1`; checks Prefect reachability when `PREFECT_API_URL` is set         |
| `POST`   | `/api/v1/experiments`                | 201    | Create an experiment record; returns `ExperimentResult` with `status=pending`                                 |
| `GET`    | `/api/v1/experiments`                | 200    | Paginated experiment list (`limit`, `offset`); deterministic order by `experiment_id`                         |
| `GET`    | `/api/v1/experiments/{id}`           | 200    | Experiment detail — 404 if not found; 422 on non-UUID path segment                                           |
| `DELETE` | `/api/v1/experiments/{id}`           | 200    | Cancel a `pending`, `queued`, or `running` experiment; 409 for terminal statuses; 404 if not found            |
| `WS`     | `/api/v1/experiments/{id}/live`      | —      | WebSocket — streams `{experiment_id, status, epoch, loss, metrics}` JSON every 2 s; `epoch` and `loss` are top-level scalar fields extracted from the stored `metrics` dict (`null` before the first runner epoch write); the full `metrics` dict is included for backward compatibility; closes on terminal status |
| `POST`   | `/api/v1/sweeps`                     | 202    | Trigger Prefect `meta_informed_sweep` flow; store sweep state; returns `{sweep_id}`                           |
| `GET`    | `/api/v1/sweeps/{id}`                | 200    | Sweep status — `{n_trials_total, n_completed, n_failed, best_result}`; 404 for unknown sweep                  |
| `GET`    | `/api/v1/sweeps/{id}/results`        | 200    | Trial results sorted by objective descending; 404 for unknown sweep                                           |
| `POST`   | `/api/v1/search-spaces`              | 201    | Persist a search space definition; returns `SearchSpaceRecord` with `search_space_id`                         |
| `GET`    | `/api/v1/search-spaces`              | 200    | Paginated list of persisted search space definitions                                                          |


**Architecture highlights:**

- **`create_app()` factory + module-level `app` instance** — `app = create_app()` is declared at module scope so that `uvicorn orcalab.api.main:app` resolves correctly at container startup. DB engine and `async_sessionmaker` are initialised once at ASGI lifespan startup and stored on `app.state`. `app.state.sweeps: dict[str, dict]` (in-memory sweep store) is also initialised at startup and disposed gracefully on shutdown.
- **Dependency injection** — `get_db` yields an `AsyncSession`; `get_experiment_repo` and `get_search_space_repo` inject typed repository instances; `get_sweeps_store` returns `app.state.sweeps`. All dependencies are overridable in tests via `dependency_overrides`.
- **CORS — deny by default** — `allow_origins=[]` and `allow_credentials=False` when the `CORS_ORIGINS` env var is not set. When set, origins are parsed from a comma-separated list.
- **Request logging** — `RequestLoggingMiddleware` logs every request with method, path, status code, and elapsed time in milliseconds. Uses `try/finally` so the log line is always written even when `call_next` raises (defaulting to status 500 in that case).
- **Atomic experiment cancellation** — `DELETE /experiments/{id}` calls `ExperimentLifecycle.transition(CANCELLED)`, which uses `repository.update_status_if_current` — a single conditional `UPDATE WHERE status = current_status`. A concurrent status change that causes zero rows to be updated raises `InvalidTransitionError`, surfaced as 409 to the caller rather than silently discarding the conflict.
- **Prefect triggering** — `POST /sweeps` POSTs to `{PREFECT_API_URL}/deployments/name/meta_informed_sweep/default/create_flow_run` via `httpx.AsyncClient`. Non-2xx responses emit a `logger.warning` without failing the request. When `PREFECT_API_URL` is not set, no HTTP call is made and `flow_run_id` is stored as `None`.
- **WebSocket metric streaming** — the `/experiments/{id}/live` handler polls the DB every 2 s using the app-level `db_sessionmaker` directly (bypassing the HTTP-only `get_db` dependency). Every message includes `epoch` and `loss` as top-level scalar fields extracted from `experiment.metrics` (both `null` before the runner's first epoch write), plus the full `metrics` dict for backward compatibility. Clients can therefore assert `"epoch" in data` and `"loss" in data` unconditionally — the fields are always present, with `null` values before training starts. The handler closes automatically on `COMPLETED`, `FAILED`, or `CANCELLED` status, or on `WebSocketDisconnect` from the client.
- **Sweeps — in-memory state** — sweep records are stored in `app.state.sweeps` (a plain dict keyed by `sweep_id`). `best_result` is computed on-read as `max(results, key=objective)`. This avoids requiring a DB migration for sweep state.

**Integration test coverage:**

The API test suite lives under `tests/integration/api/` and requires no running database, Prefect server, or MLflow instance — all external dependencies are mocked via `dependency_overrides` and `unittest.mock`.

| Test file                     | Tests | Covers                                                                                               |
|-------------------------------|-------|------------------------------------------------------------------------------------------------------|
| `test_health.py`              | 6     | Root endpoint, health ok, Prefect degraded when `PREFECT_API_URL` unset                              |
| `test_experiments.py`         | 23    | Create (201, pending status, repo call), list (pagination, limit/offset), get (200/404/422), delete (cancel, 409, 404, atomic assert) |
| `test_sweeps.py`              | 16    | POST 202, sweep_id in response, sweep stored, no Prefect call when URL unset, 422 validation, search_space stored, status 200/404, results sorted/empty/404 |
| `test_search_spaces.py`       | 10    | Create (201, search_space_id, repo call, definition forwarded, name passed), list (200, list, repo call, records, pagination) |
| `test_websocket.py`           | 15    | Accepts connection, streams metrics, closes on completed/failed, error on unknown id, experiment_id in messages, handles disconnect; `TestWebSocketSpecAssertions` (8 tests) — top-level `epoch` and `loss` fields present in every message, stored values reflected, backward-compat `metrics` dict, `null` before first write, epoch number advances across successive messages |
| `test_dockerfile.py`          | 12    | Multi-stage build structure, builder uv install, runtime venv copy, source copy, HEALTHCHECK, EXPOSE, CMD |
| `test_docker_compose.py`      | 18    | orcalab service config (env vars, depends_on, healthcheck, port); orcalab-dashboard service (port, command, ORCALAB_API_URL) |
| `test_init_prefect.py`        | 6     | Work-pool creation args (prefect work-pool create orcalab-pool --type process), check=True |
| `test_app_module_export.py`   | 9     | Module-level app attribute exists, is FastAPI instance, has correct title, all route prefixes registered |
| **Total**                     | **115**|                                                                                                     |

---

### CLI (`orcalab`)

Four commands installed as the `orcalab` entry point.

```bash
orcalab --help           # List all commands
orcalab <command> --help # Per-command usage
```


| Command     | Purpose                                              | Key Options                                           |
| ----------- | ---------------------------------------------------- | ----------------------------------------------------- |
| `init`      | Create workspace directories and default config      | —                                                     |
| `sweep`     | Run a hyperparameter sweep for a given task          | `--n-trials INT`, `--strategy tpe\|cma\|random`       |
| `serve`     | Start the FastAPI service on port 8001               | `--host TEXT`, `--port INT`, `--reload`               |
| `dashboard` | Launch the Streamlit dashboard on port 8502          | `--port INT`                                          |


Port defaults (8001 for the API, 8502 for the dashboard) are chosen to avoid collisions with OrcaMind (8000, 8501).

### Hydra Configuration (`config/`)

```text
config/
├── config.yaml          # Root: prefect.api_url, orcamind.api_url, resources
├── search/
│   └── bayesian.yaml    # TPE sampler: n_startup_trials=10, n_ei_candidates=24, multivariate=true
└── pruner/
    └── asha.yaml        # ASHA: min_resource=1, max_resource=100, reduction_factor=3
```

All external service URLs (Prefect API, OrcaMind API) are resolved via `${oc.env:VAR,default}` interpolation — no credentials appear in committed config files.

---

## `orcanet` — Cross-Domain Knowledge Transfer Agent

OrcaNet is the third component of the Orca ecosystem. It orchestrates OrcaMind and OrcaLab to deliver end-to-end cross-domain knowledge transfer: retrieving proven model configurations from one domain and adapting them to a new target task, validated through a real OrcaLab experiment. All namespaces are fully implemented: embeddings (DANN, text, GNN-based architecture embedders), transfer (CKA feature transfer, weight transfer, architecture transfer, multi-task transfer), retrieval (FAISS vector search, metadata filtering, LLM re-ranking), reasoning (LangChain ReAct agent with retry and structured output), and the FastAPI HTTP service on port 8002.

### Package Structure

```text
orcanet/
├── embeddings/    # CrossDomainEmbedder (DANN, implemented), TextTaskEmbedder (sentence-transformers + stats fusion, implemented), ArchitectureEmbedder (GNN-based, implemented)
├── transfer/      # CKA feature transfer, weight transfer, architecture transfer, multi-task transfer (all implemented)
├── retrieval/     # QueryExpander, LLMRanker, HybridRetriever — three-stage async pipeline (implemented)
├── reasoning/     # LangChain ReAct agent, Pydantic response models, retry logic (implemented)
│   └── prompts/   # Prompt templates: transfer explanation, task similarity, architecture recommendation
├── api/           # FastAPI service — 8 live endpoints on port 8002 (main.py, deps.py, schemas.py, middleware.py, routers/)
└── cli.py         # Typer CLI — serve and version commands
```

### Module Namespaces

#### `orcanet.embeddings` — Domain-Invariant Embedders

The embeddings namespace houses three implemented embedders: the DANN-based cross-domain embedder, the text-based task description embedder, and the GNN-based architecture embedder.

##### `CrossDomainEmbedder` (implemented)

A Domain-Adversarial Neural Network (Ganin et al. 2016) that learns 64-dimensional task representations invariant to the domain the data was drawn from. It is the embedding backbone that the OrcaNet retrieval and transfer layers consume. All defaults match `config/embedder/cross_domain.yaml` so Hydra-constructed and in-code instances are identical.

**Architecture:**

```
Statistical meta-features (25-dim)
              ↓
    _FeatureMLP  [Linear → BatchNorm1d → ReLU] × n_hidden  +  Linear
              ↓
        64-dim features (unnormalised)
         ↙                      ↘
task_classifier           domain_classifier
Linear(64 → 3)            GRL → Linear(64,64) → ReLU → Linear(64, 10)
```

**Classes:**

| Class | Role |
|---|---|
| `GradientReversalFunction` | Custom `torch.autograd.Function`: identity in the forward pass; returns `−alpha × grad_output` in the backward pass so upstream gradients are negated and scaled |
| `GradientReversalLayer` | `nn.Module` wrapper around `GradientReversalFunction` with a mutable `alpha` attribute. Stored as `self._grl` on the embedder so the training loop can update `alpha` in-place; the `domain_classifier` Sequential holds a reference to the same object, so updates propagate automatically |
| `_FeatureMLP` | Private feed-forward backbone: `Linear → BatchNorm1d → ReLU` per hidden layer, final `Linear`. Deliberately omits L2 normalisation — that step is applied externally by `embed()` so that `task_classifier` and `domain_classifier` receive raw, unnormalised features |
| `CrossDomainEmbedder` | Main model; wires together `_FeatureMLP`, the task-type classification head, and the adversarial domain head |

**Constructor:**

```python
CrossDomainEmbedder(
    input_dim: int = 25,               # StatisticalEmbedder output dimensionality
    embedding_dim: int = 64,           # Shared feature space dimensionality
    n_domains: int = 10,               # Domain classifier output classes
    n_task_types: int = 3,             # Task-type classifier output classes
    hidden_dims: list[int] | None = None,  # Defaults to [128, 64]
)
```

**Public API:**

| Method | Signature | Description |
|---|---|---|
| `forward` | `(x: Tensor) → tuple[Tensor, Tensor, Tensor]` | Returns `(features, task_logits, domain_logits)` — raw, unnormalised. Use during training only. |
| `embed` | `(x: Tensor) → Tensor` | Returns L2-normalised feature embeddings for cosine-similarity retrieval. Temporarily switches to eval mode and disables gradients, then restores the caller's original training state in a `finally` block so the method is safe to call from inside a training loop. |
| `fit` | `(x, task_labels, domain_labels, epochs=20, lr=1e-3, domain_lambda=1.0) → dict[str, list[float]]` | Trains with the DANN objective using Adam and the Ganin et al. progressive alpha schedule. Returns `{"task_loss": [...], "domain_loss": [...]}` with one float per epoch. |
| `save` | `(path: str \| Path) → None` | Serialises constructor config and `state_dict()` via `torch.save`. Checkpoint contains only Python primitives and tensors, safe to load with `weights_only=True`. |
| `load` | `(path: str \| Path) → CrossDomainEmbedder` *(classmethod)* | Reconstructs the model from a checkpoint and returns it in eval mode. |

**Training details — `fit()` :**

The DANN objective combines two cross-entropy losses in a single forward pass:

```
L_total = L_task + λ · L_domain
```

`L_task` supervises the task-type head normally. `L_domain` is computed through the GRL, so its gradient arrives at the feature extractor with the sign reversed — the extractor is penalised for *successfully* predicting domain, which is what drives domain invariance.

GRL alpha increases from 0 to 1 following the schedule from Ganin et al.:

```
p  = (epoch − 1) / max(epochs − 1, 1)   # training progress ∈ [0, 1]
α  = 2 / (1 + exp(−10p)) − 1            # sigmoid-shaped ramp
```

At epoch 1, α ≈ 0, giving the task head time to establish useful representations before adversarial pressure builds. By the final epoch α = 1.0 and the domain discriminator is fully engaged.

**Usage example:**

```python
from orcanet.embeddings import CrossDomainEmbedder
import torch

model = CrossDomainEmbedder()                     # defaults: 25→64 dim, 10 domains, 3 task types

# Training
x             = torch.randn(128, 25)              # N × input_dim
task_labels   = torch.randint(0, 3, (128,))       # 0=classification, 1=regression, 2=time_series
domain_labels = torch.randint(0, 10, (128,))      # source domain index

history = model.fit(x, task_labels, domain_labels, epochs=50, lr=1e-3)
# history = {"task_loss": [...50 floats], "domain_loss": [...50 floats]}

# Inference — L2-normalised, eval mode, no gradient tracking
embeddings = model.embed(x)                       # shape (128, 64), unit vectors

# Persistence
model.save("/data/models/cross_domain_embedder.pt")
restored = CrossDomainEmbedder.load("/data/models/cross_domain_embedder.pt")
```

**Lazy import pattern:**

`orcanet.embeddings.__init__` uses a module-level `__getattr__` shim to defer `import torch` until `CrossDomainEmbedder`, `GradientReversalLayer`, `TextTaskEmbedder`, `ArchitectureEmbedder`, or `ArchitectureGraph` is accessed by name. Importing the namespace (`import orcanet.embeddings`) carries zero PyTorch startup cost, which keeps the service importability test and cold-start time unaffected.


##### `TextTaskEmbedder` (implemented)

A plain Python (non-`nn.Module`) class that encodes natural language task descriptions into 384-dim semantic vectors via `sentence-transformers` and optionally fuses them with 25-dim statistical meta-features from `StatisticalEmbedder`. The fused embedding serves as a richer task fingerprint for retrieval and transfer scoring when a text description of the target task is available.

**Architecture:**

```
Natural language description (string)
              |
    SentenceTransformer (all-MiniLM-L6-v2)
              |
       384-dim text vector (L2-normalised)
              |
     +---------+----------+
     |                    |
text_vec (384)       stat_vec (25)
     |          <-- StatisticalEmbedder
     +-----------+--------+
                 |
    Fusion  (concat / add / attention)
                 |
      output_dim-dim fused vector (L2-normalised)
```

**Fusion strategies:**

| Strategy | Module | Mechanism |
|---|---|---|
| `"concat"` | `nn.Sequential` | Concatenates the 384-dim text vector and 25-dim stat vector into a 409-dim input; passes through `Linear(409, 256) → ReLU → Linear(256, output_dim)` |
| `"add"` | `_AddFusion` | Two independent linear projections (`text_proj` and `stat_proj`, both `Linear(·, output_dim)`) summed element-wise |
| `"attention"` | `_AttentionFusion` | Same two independent projections; gating weights computed via `softmax(Linear(2·output_dim, 2))` applied to the concatenation; weighted sum |

All fusion networks are initialised with random weights and placed in eval mode. They are not pre-trained — the embedder is designed for use as a feature extractor whose fusion weights can be fine-tuned downstream by OrcaNet’s training pipeline.

**Constructor:**

```python
TextTaskEmbedder(
    model_name: str = "all-MiniLM-L6-v2",                # SentenceTransformer model identifier
    statistical_embedder: StatisticalEmbedder | None = None,  # instantiated fresh if None
    fusion: str = "concat",                               # "concat", "add", or "attention"
    output_dim: int = 128,                                # fused embedding dimensionality
)
```

Raises `ValueError` (message: `"fusion must be one of ..."`) if `fusion` is not one of the three supported values.

**Public API:**

| Method | Signature | Description |
|---|---|---|
| `embed_from_description` | `(description: str) → np.ndarray` | Encodes *description* via SentenceTransformer; L2-normalises; returns shape `(384,)`. No statistical features consumed. |
| `embed_with_stats` | `(description: str, dataset: pd.DataFrame, labels: pd.Series \| None = None) → np.ndarray` | Fuses text and statistical embeddings via the configured fusion net inside a `torch.no_grad()` context; L2-normalises; returns shape `(output_dim,)`. |
| `embed_batch_descriptions` | `(descriptions: list[str]) → np.ndarray` | Encodes all descriptions in a single batched SentenceTransformer call (`batch_size=32`); row-wise L2-normalises; returns shape `(N, 384)`. |

**Usage example:**

```python
from orcanet.embeddings import TextTaskEmbedder
import pandas as pd
import numpy as np

embedder = TextTaskEmbedder(fusion="concat", output_dim=128)

# Text-only embedding
vec = embedder.embed_from_description("binary image classification with augmented training set")
# vec.shape == (384,), L2-normalised

# Fused embedding with dataset statistics
X = pd.DataFrame({"feature_1": [1.0, 2.0], "feature_2": [3.0, 4.0]})
y = pd.Series([0, 1])
fused = embedder.embed_with_stats("binary classification", X, y)
# fused.shape == (128,), L2-normalised

# Batch encoding
descriptions = ["image classification", "time series regression", "NLP sentence similarity"]
matrix = embedder.embed_batch_descriptions(descriptions)
# matrix.shape == (3, 384)

# Cosine similarity (L2-normalised vectors → dot product == cosine)
similarity = float(np.dot(matrix[0], matrix[1]))
```

##### `ArchitectureEmbedder` (implemented)

A GNN-based `nn.Module` that encodes any model architecture config dict into a 32-dimensional L2-normalised embedding by representing the architecture as a graph — layers as nodes, connectivity as edges — and applying residual adjacency-matrix message passing. The resulting unit vectors support cosine-similarity retrieval over architecture libraries, enabling the OrcaNet transfer pipeline to identify structurally similar source architectures without manual curation.

**Architecture:**

```
ModelConfig dict  {"layers": [...], "skip_connections": [...]}
              ↓
    ArchitectureGraph.from_model_config()
              ↓
   node_features (n_nodes × 16)   edge_index (2 × n_edges)
              ↓
    node_encoder: Linear(16 → 64) + ReLU
              ↓
    3× residual message passing: h ← h + W(Â h)
         (Â = row-normalised adjacency with self-loops, undirected)
              ↓
    mean pool across nodes → (64,)
              ↓
    readout: Linear(64 → 32)
              ↓
    L2 normalise → (32,) unit vector
```

**`ArchitectureGraph` dataclass:**

`from_model_config(config)` converts a `ModelConfig` dict into GNN-ready numpy arrays:

| Field | Shape | dtype | Content |
|---|---|---|---|
| `node_features` | `(n_nodes, 16)` | float32 | 8-dim one-hot layer type + 1 log-scaled size + 7-dim one-hot activation |
| `edge_index` | `(2, n_edges)` | int64 | `[sources, targets]` — sequential `i → i+1` edges plus validated skip connections |
| `graph_features` | `(3,)` | float32 | Global statistics: `[log1p(Σsize), depth, log1p(max_width)]` |

Recognised layer types (indices 0–7): `linear`, `conv2d`, `lstm`, `attention`, `batchnorm`, `dropout`, `pooling`, `embedding`. Recognised activations (indices 0–6): `relu`, `sigmoid`, `tanh`, `gelu`, `selu`, `softmax`, `none`. Unknown types produce all-zero one-hot slots. Layer sizes are normalised as `log1p(size) / log1p(4096)` so the scalar feature stays in `[0, 1]` and does not dominate cosine similarity.

Input sanitisation: negative or non-numeric `size` values are clamped to 0 with a debug log rather than propagating a `ValueError`; malformed skip-connection entries (wrong length, non-numeric elements, non-iterable) are skipped per-entry with a debug log. An empty or missing `layers` key produces a single zero-node with no edges and zero graph features rather than raising.

**Constructor:**

```python
ArchitectureEmbedder(
    node_dim:   int = 16,   # must match _NODE_DIM = len(layer_types) + 1 + len(activation_types)
    hidden_dim: int = 64,   # GNN hidden width
    output_dim: int = 32,   # output embedding dimensionality
)
```

**Public API:**

| Method | Signature | Description |
|---|---|---|
| `embed` | `(config: ModelConfig) → np.ndarray` | Returns a `(output_dim,)` float32 unit vector. Switches to eval + `torch.no_grad()`, restores the caller's training mode in a `finally` block so the method is safe to call from inside a training loop. |
| `similarity` | `(config_a, config_b: ModelConfig) → float` | Cosine similarity in `[-1, 1]`; 1.0 for identical configs. Implemented as a dot product of unit vectors — no separate normalisation step needed. |
| `find_similar_architectures` | `(query, candidates, top_k=5) → list[tuple[ModelConfig, float]]` | Ranks `candidates` by cosine similarity to `query`; returns the top-`top_k` `(config, score)` pairs sorted descending. Raises `ValueError` for negative `top_k`; `top_k=0` returns an empty list. |

**Message-passing details:**

The default path (no `torch-geometric`) builds a dense row-normalised adjacency matrix `Â` with self-loops, treats edges as undirected, and runs three rounds of residual propagation: `h ← h + W(Âh)`. Residuals preserve each node's own identity after neighbourhood aggregation — without them, iterated averaging collapses all sequential-graph embeddings toward a common direction regardless of layer types. If `torch-geometric` is installed (`pip install orcanet[gnn]`), `GCNConv` layers replace the dense adjacency path transparently; the public API is unchanged.

All tensor allocations in `_forward_graph` are dispatched to `self.node_encoder.weight.device`, and `_build_adj_norm` inherits the device from the `edge_index` tensor, so moving the module to CUDA with `.to("cuda")` works without modification.

**Usage example:**

```python
from orcanet.embeddings import ArchitectureEmbedder, ArchitectureGraph

embedder = ArchitectureEmbedder()          # defaults: 16-dim nodes → 64-dim hidden → 32-dim output

mlp_config = {
    "layers": [
        {"type": "linear", "size": 512, "activation": "relu"},
        {"type": "linear", "size": 256, "activation": "relu"},
        {"type": "linear", "size": 10,  "activation": "softmax"},
    ]
}
cnn_config = {
    "layers": [
        {"type": "conv2d",  "size": 64, "activation": "relu"},
        {"type": "pooling", "size": 64},
        {"type": "linear",  "size": 10, "activation": "softmax"},
    ],
    "skip_connections": [[0, 2]]           # optional: adds an edge from layer 0 to layer 2
}

# Embed a single architecture → (32,) unit vector
vec = embedder.embed(mlp_config)

# Cosine similarity between two architectures
sim = embedder.similarity(mlp_config, cnn_config)      # float in [-1, 1]

# Ranked retrieval over a candidate pool
candidates = [mlp_config, cnn_config, ...]
results = embedder.find_similar_architectures(mlp_config, candidates, top_k=3)
# → [(config, score), ...]  sorted by score descending

# Direct graph inspection
graph = ArchitectureGraph.from_model_config(mlp_config)
# graph.node_features  shape (3, 16)  — one row per layer
# graph.edge_index     shape (2, 2)   — two sequential edges
# graph.graph_features shape (3,)     — [log1p(778), 3.0, log1p(512)]
```

#### `orcanet.transfer` — Transfer Strategies

The transfer subpackage provides three concrete transfer strategies sharing a common ABC and return type:

- **`FeatureTransfer`** — scores transferability via per-layer linear Centered Kernel Alignment (Kornblith et al. 2019) computed over forward-hook activations on shared probe data; selectively patches weights for high-CKA layers.
- **`WeightTransfer`** — matches parameter tensors directly by name, shape, or both; deep-copies the source architecture as the transfer base and reinitialises unmatched tensors; pairs with `get_optimizer_with_layer_lr` for layer-wise learning-rate decay.
- **`ArchitectureTransfer`** — recommends and adapts architectures for a target domain by comparing architecture graph embeddings; retrieves the source task's best-known architecture from OrcaMind, scores all locally registered candidate configs by cosine similarity, and builds an adapted `nn.Sequential` model with correct input/output dimensions.

##### `TransferStrategy` (`base.py`)

Abstract base class that every transfer strategy must implement.

```python
class TransferStrategy(ABC):
    @abstractmethod
    def score_transfer(self, source: Task, target: Task) -> TransferScore: ...
    @abstractmethod
    def execute_transfer(self, source: Task, target: Task, source_model: nn.Module) -> nn.Module: ...
    @abstractmethod
    def get_transfer_metadata(self) -> dict: ...
```

##### `TransferScore` (`types.py`)

Rich internal dataclass representing the output of a scoring run. Distinct from `orca_shared.schemas.transfer.TransferScore`, which is the lightweight API schema for inter-service exchange.

| Field | Type | Description |
|---|---|---|
| `overall` | `float` | Depth-weighted mean CKA across all scored layers, clipped to [0, 1] |
| `layer_scores` | `dict[str, float]` | Per-named-layer CKA value |
| `recommended_layers` | `list[str]` | Layers with CKA above `cka_threshold` |
| `reasoning` | `str` | Human-readable summary for logging and agent consumption |

##### `linear_cka` / `FeatureTransfer` (`feature_transfer.py`)

`linear_cka(X, Y)` — pure-numpy function implementing the Kornblith et al. 2019 linear CKA formula:

```
CKA = ||Y_c^T X_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)
```

where the subscript *c* denotes column-mean centering. Returns a float in [0, 1], clamped via `min(..., 1.0)` to absorb floating-point noise.

`FeatureTransfer` — concrete `TransferStrategy` implementing CKA-based scoring and selective weight patching.

**Workflow:**

1. Register source and target `nn.Module` instances via `register_model(task_id, model)`.
2. Provide `probe_data` — an `(n_samples, input_dim)` float32 array shared by both models.
3. Call `score_transfer(source, target)` to collect per-layer activations via PyTorch forward hooks, compute layer-wise CKA, and return a `TransferScore`.
4. Optionally call `execute_transfer(source, target, source_model)` to clone the registered target model and patch weights from recommended layers.

**Key implementation details:**

- **Forward hook collection** — `register_forward_hook` is attached to every named submodule (root wrapper excluded). Hooks capture `output.detach().cpu().numpy()` for any tensor output with `ndim >= 2`. Hooks and training mode are always restored in a `try/finally` block.
- **Device handling** — the probe tensor is moved to each model's device (`next(model.parameters()).device`) before the forward pass, supporting CPU, CUDA, and MPS models.
- **Depth-ordered scoring** — common layers are sorted by `(name.count("."), name)` so shallower layers (fewer dots = closer to model root) appear first, consistent with the depth-weighted averaging step.
- **Depth-weighted averaging** — layer at index `i` receives weight `1/(i+1)`, normalised to sum 1. Shallower layers get higher weight because they encode more broadly transferable features.
- **Weight patching** — `execute_transfer` clones the registered target model (not the source), then copies matching-shape parameters from `source_model` for every `recommended_layers` entry, preserving target-specific capacity in non-recommended layers.

```python
from orcanet.transfer import FeatureTransfer, TransferScore

transfer = FeatureTransfer(probe_data=probe, cka_threshold=0.5)
transfer.register_model(str(source_task.task_id), source_model)
transfer.register_model(str(target_task.task_id), target_model)

score: TransferScore = transfer.score_transfer(source_task, target_task)
# score.overall       — depth-weighted mean CKA (float in [0, 1])
# score.layer_scores  — {'0': 0.94, '1': 0.87, ...}
# score.recommended_layers  — layers above cka_threshold=0.5

adapted = transfer.execute_transfer(source_task, target_task, source_model)
# adapted is a clone of the registered target model with source weights patched in
```

##### `WeightTransfer` / `get_optimizer_with_layer_lr` (`weight_transfer.py`)

`WeightTransfer` — concrete `TransferStrategy` that transfers parameter tensors directly, without requiring probe data or forward passes.

**Workflow:**

1. Register source and target `nn.Module` instances via `register_model(task_id, model)`.
2. Call `score_transfer(source, target)` to compute per-parameter match flags and an overall ratio.
3. Call `execute_transfer(source, target, source_model)` to produce the adapted model; transferred parameter names are stored in `last_transferred`.
4. Pass `transfer.last_transferred` to `get_optimizer_with_layer_lr` to build a learning-rate–stratified Adam optimizer.

**`match_by` modes:**

| Mode | Match condition | Use case |
|---|---|---|
| `"name"` (default) | Parameter name exists in source state dict | Same architecture, different initialisation |
| `"shape"` | Any source tensor has the same shape | Architecture families with shared tensor dimensions |
| `"both"` | Name present **and** shapes agree | Cross-architecture transfer where last-layer sizes differ |

**Key implementation details:**

- **Deepcopy base** — `execute_transfer` starts from `deepcopy(self._model_registry[target_id])`, so the result preserves the target architecture and capacity. The source weights are copied in on top for matched layers only.
- **Shape-safe copy resolution** — `_find_source_tensor` is the sole entry point for resolving which source tensor to copy. It always verifies shape compatibility before returning, so `copy_()` is never called with mismatched shapes regardless of `match_by` mode. A `None` return triggers reinitialisation.
- **Safe reinitialisation** — `_safe_reinit` applies `nn.init.kaiming_uniform_` to 2-D+ tensors (weight matrices, convolutional kernels) and `nn.init.zeros_` to 1-D tensors (bias vectors). This avoids the `ValueError` that `kaiming_uniform_` raises on 1-D inputs.
- **`last_transferred` attribute** — `execute_transfer` returns `nn.Module` and stores the list of transferred parameter names in `self.last_transferred`. Pass this list to `get_optimizer_with_layer_lr` without needing to re-run `score_transfer`.
- **Binary scoring** — `score_transfer` produces `layer_scores = {name: 1.0 | 0.0}` rather than a continuous similarity value. The `overall` field is the exact matched-parameter ratio: `n_matched / n_total`.

`get_optimizer_with_layer_lr(model, transferred_layers, base_lr, decay=0.1)` — module-level function that creates one `{"params": [p], "lr": ...}` group per named parameter. Transferred parameters receive `base_lr * decay`; all others receive `base_lr`. Returns a `torch.optim.Adam` ready for training.

```python
from orcanet.transfer import WeightTransfer, get_optimizer_with_layer_lr

transfer = WeightTransfer(match_by="both", layer_lr_decay=0.1)
transfer.register_model(str(source_task.task_id), source_model)
transfer.register_model(str(target_task.task_id), target_model)

score: TransferScore = transfer.score_transfer(source_task, target_task)
# score.overall       — matched / total (float in [0, 1])
# score.layer_scores  — {'0.weight': 1.0, '0.bias': 1.0, '2.weight': 0.0, '2.bias': 0.0}
# score.recommended_layers — ['0.weight', '0.bias']

adapted = transfer.execute_transfer(source_task, target_task, source_model)
# adapted — deepcopy of registered target model with source weights patched in
# transfer.last_transferred — ['0.weight', '0.bias']  (parameter names that were copied)

optimizer = get_optimizer_with_layer_lr(adapted, transfer.last_transferred, base_lr=1e-3, decay=0.1)
# transferred params get lr=1e-4; reinitialised params get lr=1e-3
```

##### `ArchitectureTransfer` / `adapt_architecture` (`architecture_transfer.py`)

`ArchitectureTransfer` — concrete `TransferStrategy` that recommends and adapts architectures for a target domain using architecture graph-embedding similarity instead of weight-level comparison.

**Workflow:**

1. Register candidate architecture configs with `register_config(name, config)`.
2. Call `score_transfer(source, target)` to fetch the source task's best architecture from OrcaMind and score every registered candidate by cosine similarity.
3. Call `execute_transfer(source, target, source_model)` to build and return an `nn.Sequential` adapted to the target task's input/output dimensions.

**Config format (`ArchConfig`):**

```python
{
    "input_dim": 128,
    "layers": [
        {"type": "linear", "size": 256, "activation": "relu"},
        {"type": "linear", "size": 64,  "activation": "relu"},
        {"type": "linear", "size": 10,  "activation": "none"},
    ]
}
```

Recognised activation values: `"relu"`, `"sigmoid"`, `"tanh"`, `"gelu"`, `"none"` (no activation module appended). Unknown activations are silently ignored.

**Key implementation details:**

- **OrcaMind lookup** — `score_transfer` calls `orcamind_client.get_best_model(source.task_id)` asynchronously to identify the source architecture name; the name is used to look up the source config from the local registry.
- **Sync/async bridge** — `_run_coro()` handles the impedance mismatch between the synchronous `TransferStrategy` ABC and the async `OrcaMindClient`. When called from inside a running event loop it spawns a background `threading.Thread` with its own loop to avoid re-entrance.
- **`adapt_architecture(config, target_task)`** — pure function that deep-copies the config and overwrites `input_dim` with `target_task.n_features` (if not `None`) and the last layer's `size` with `target_task.n_classes` (if not `None`). Hidden layers are untouched.
- **Middle-layer weight copying** — `execute_transfer` copies parameters from `source_model` into matching middle positions (skipping the first input layer and the last output layer) where shapes agree. Shape mismatches are silently skipped; all linear layers are initialised with `kaiming_uniform_` (weights) and `zeros_` (biases) before the copy step.
- **`recommended_layers` is always `[]`** — architecture transfer is a model-level decision; there is no per-layer selection concept.

**Constructor:**

```python
ArchitectureTransfer(
    architecture_embedder: ArchitectureEmbedder,
    orcamind_client: OrcaMindClient,
    top_k_candidates: int = 10,
)
```

**Public API:**

| Method | Signature | Description |
|---|---|---|
| `register_config` | `(name: str, config: ArchConfig) -> None` | Store a named architecture config in the local registry |
| `score_transfer` | `(source: Task, target: Task) -> TransferScore` | Fetch source architecture from OrcaMind; score all registered candidates; return best match |
| `execute_transfer` | `(source: Task, target: Task, source_model: nn.Module) -> nn.Module` | Adapt best config for target task; build `nn.Sequential`; copy middle-layer weights from source |
| `get_transfer_metadata` | `() -> dict` | Return `{"strategy": "architecture_transfer", "top_k_candidates": int, "n_registered_configs": int}` |

`adapt_architecture(config, target_task)` — module-level helper also exported from `orcanet.transfer`. Deep-copies the config and updates the boundary dimensions, leaving hidden layers intact.

```python
from orcanet.transfer import ArchitectureTransfer, adapt_architecture

transfer = ArchitectureTransfer(
    architecture_embedder=embedder,
    orcamind_client=client,
    top_k_candidates=10,
)

transfer.register_config("mlp_128_64", {
    "input_dim": 25,
    "layers": [
        {"type": "linear", "size": 128, "activation": "relu"},
        {"type": "linear", "size": 64,  "activation": "relu"},
        {"type": "linear", "size": 10,  "activation": "none"},
    ]
})

score = transfer.score_transfer(source_task, target_task)
# score.overall            — highest cosine similarity across candidates
# score.layer_scores       — {"mlp_128_64": 0.87}
# score.recommended_layers — []  (architecture-level decision)

adapted = transfer.execute_transfer(source_task, target_task, source_model)
# adapted: nn.Sequential with in_features == target_task.n_features
#           and out_features == target_task.n_classes;
#           hidden-layer weights copied from source_model where shapes match
```

##### `MultiTaskTransfer` and `MultiTaskModel` (implemented)

Joint training across multiple related tasks using a shared backbone and task-specific heads, with three task-weighting schemes. The strategy follows the same `TransferStrategy` interface as the other transfer implementations, returning a `TransferScore` from `score_transfer` and an `nn.Module` from `execute_transfer`.

**`_get_backbone_out_dim(backbone: nn.Module) -> int`** (module-level helper)

Infers the backbone's output dimensionality by scanning all sub-modules and returning the `out_features` of the last `nn.Linear` encountered. Raises `ValueError` if no `nn.Linear` is found. Used by `MultiTaskTransfer.__init__` so head construction is automatic regardless of backbone architecture.

**`MultiTaskModel(nn.Module)`**

The model returned by `MultiTaskTransfer.execute_transfer`. Holds a shared backbone and a per-task head registry.

| Attribute | Type | Description |
|---|---|---|
| `backbone` | `nn.Module` | Shared feature extractor, registered as a submodule |
| `task_heads` | `nn.ModuleDict` | Per-task output heads, keyed by `str(task.task_id)`. Using `nn.ModuleDict` (not a plain `dict`) ensures heads appear in `model.parameters()` and `model.state_dict()` so they are optimised and serialised correctly. |
| `task_weighting` | `str` | The weighting scheme this model was constructed with (`"equal"`, `"uncertainty"`, or `"gradnorm"`). Stored for reference — does not affect `forward`. |
| `log_sigmas` | `nn.ParameterDict` | Learnable log-variance scalars for uncertainty weighting, one per task id. Empty for `"equal"` and `"gradnorm"` schemes. Using `nn.ParameterDict` ensures they are tracked by the optimiser. |

Methods:

| Method | Signature | Description |
|---|---|---|
| `forward` | `(x: Tensor, task_id: str) → Tensor` | Passes `x` through `backbone`, then routes the resulting features through `task_heads[task_id]`. Raises `KeyError` for unregistered task ids. |
| `compute_loss` | `(batch: dict[str, tuple[Tensor, Tensor]], weights: dict[str, float]) → Tensor` | Computes a weighted sum of per-task cross-entropy losses: `Σ weights[tid] · CE(forward(x, tid), y)`. Used for `"equal"` and `"gradnorm"` weighting. Returns a scalar tensor with a `grad_fn`. |
| `compute_uncertainty_loss` | `(batch: dict[str, tuple[Tensor, Tensor]]) → Tensor` | Implements the Kendall et al. 2018 objective: `L = Σ exp(−2·log_σᵢ) · CEᵢ + log_σᵢ`. Tasks with irreducible noise learn large `log_σᵢ` (lower effective weight); clean tasks maintain small `log_σᵢ` (higher effective weight). The regularisation term `log_σᵢ` prevents all sigmas from growing to infinity. Returns a scalar tensor — gradients flow into `log_sigmas` automatically. |

**`MultiTaskTransfer(TransferStrategy)`**

Constructor:

```python
MultiTaskTransfer(
    backbone: nn.Module,
    task_weighting: str = "equal",        # "equal" | "uncertainty" | "gradnorm"
    task_head_hidden_dim: int = 64,
    embedder: CrossDomainEmbedder | None = None,
)
```

Raises `ValueError` on construction if `task_weighting` is not one of the three supported values.

| Attribute | Description |
|---|---|
| `backbone` | Shared feature extractor passed at construction; also the backbone embedded in every `MultiTaskModel` returned by `execute_transfer`. |
| `_backbone_out_dim` | Inferred once at construction from the last `nn.Linear` in `backbone`. |
| `_task_heads` | Plain `dict[str, nn.Module]` maintained by the strategy. Becomes `task_heads` in `MultiTaskModel`. |
| `_task_weights` | Per-task float weights computed by `_update_weights()` after every `add_task` call. |
| `_log_sigmas` | Per-task `nn.Parameter(torch.zeros(1))` created by `add_task` when `task_weighting == "uncertainty"`. Passed into `MultiTaskModel` at `execute_transfer` time. |
| `_task_features` | Per-task feature tensors registered via `register_task_features`, used by `score_transfer`. |

Methods:

**`add_task(task: Task, head_output_dim: int) -> None`**

Creates a two-layer output head `nn.Sequential(Linear(backbone_out, hidden), ReLU(), Linear(hidden, head_output_dim))` and registers it under `str(task.task_id)`. For `"uncertainty"` weighting, also creates a `nn.Parameter(torch.zeros(1))` log-sigma for that task. Calls `_update_weights()` after registration so weights always reflect the current number of tasks.

Raises `ValueError` if `task.task_id` has already been registered. This prevents a second `add_task` call from silently discarding any trained head parameters.

**`register_task_features(task_id: str, features: Tensor) -> None`**

Stores a meta-feature tensor for a task (expected shape `(1, input_dim)` matching the embedder's `input_dim`, default 25). Follows the same pre-registration pattern as `FeatureTransfer.register_model` and `WeightTransfer.register_model`. Call this before `score_transfer` for a meaningful similarity score. The stored tensor is always detached from any autograd graph, so calling this inside a training loop does not cause the intermediate activations to be retained in memory.

**`score_transfer(source: Task, target: Task) -> TransferScore`**

If task feature tensors have been registered for both tasks via `register_task_features`, passes them through `CrossDomainEmbedder.embed()` (which returns L2-normalised vectors) and computes cosine similarity as the dot product. Returns `overall = clamp(similarity, 0, 1)`. Falls back to a neutral `overall = 0.5` with a descriptive `reasoning` string when features are not registered.

`reasoning` string semantics:

| Condition | `reasoning` |
|---|---|
| Similarity > 0.5 | `"Multi-task training beneficial: similarity {:.2f} > threshold 0.5"` |
| Similarity ≤ 0.5 | `"Multi-task training marginal: similarity {:.2f} <= threshold 0.5"` |
| No features registered | `"No task features registered for similarity computation."` |

**`execute_transfer(source: Task, target: Task, source_model: nn.Module) -> nn.Module`**

Auto-registers any task that was not pre-registered via `add_task`, using `task.n_classes` (falling back to `1`) as the head output dimension. Returns a `MultiTaskModel` with:

- `backbone = self.backbone`
- `task_heads = dict(self._task_heads)` (all tasks registered at execution time)
- `task_weighting = self.task_weighting`
- `log_sigmas = dict(self._log_sigmas)` for uncertainty weighting; `None` otherwise

`source_model` is accepted for API compatibility with the `TransferStrategy` interface but is not directly used during joint-training setup — the backbone is the shared model. The returned `MultiTaskModel` can be trained end-to-end with any standard optimiser.

**`update_gradnorm_weights(grad_norms: dict[str, float]) -> None`**

Renormalises `_task_weights` for the task ids present in `grad_norms`. Only those tasks are updated; all other registered tasks retain their existing weight exactly. The updated tasks are scaled so that their combined weight equals what they held before the call, keeping the global weight sum stable. The caller is responsible for computing gradient norms (e.g. via `torch.autograd.grad` on the backbone's last layer). Raises `ValueError` if any key in `grad_norms` does not match a registered task id. If an empty dict is passed, weights are unchanged.

**`task_weights` property**

Returns a copy of `_task_weights` as `dict[str, float]`. Pass this dict directly to `MultiTaskModel.compute_loss(batch, weights)`.

**`get_transfer_metadata() -> dict`**

Returns `{"strategy": "multi_task_transfer", "task_weighting": str, "task_head_hidden_dim": int, "n_registered_tasks": int, "backbone_out_dim": int}`.

**Weighting scheme comparison:**

| Scheme | `_task_weights` behaviour | Loss method to use | Best for |
|---|---|---|---|
| `"equal"` | Uniform `1/n` after each `add_task` | `compute_loss(batch, weights)` | Similar-difficulty tasks, baseline |
| `"uncertainty"` | Placeholder `1.0` each; effective weights via `log_sigmas` | `compute_uncertainty_loss(batch)` | Tasks with different noise levels; the model self-calibrates |
| `"gradnorm"` | Uniform init; caller calls `update_gradnorm_weights(grad_norms)` | `compute_loss(batch, weights)` | Tasks with unstable gradient magnitudes |

**Usage example:**

```python
import torch
import torch.nn as nn
from orcanet.transfer import MultiTaskTransfer, MultiTaskModel

# 1. Define a shared backbone
backbone = nn.Sequential(nn.Linear(25, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())

# 2. Construct strategy with uncertainty weighting
strategy = MultiTaskTransfer(backbone, task_weighting="uncertainty", task_head_hidden_dim=32)

# 3. Register tasks with their head output dimensions
strategy.add_task(source_task, head_output_dim=3)   # 3-class source
strategy.add_task(target_task, head_output_dim=5)   # 5-class target

# 4. Optionally register task features for score_transfer
strategy.register_task_features(str(source_task.task_id), torch.randn(1, 25))
strategy.register_task_features(str(target_task.task_id), torch.randn(1, 25))

score = strategy.score_transfer(source_task, target_task)
print(score.overall)    # e.g. 0.73 — cosine similarity of DANN embeddings
print(score.reasoning)  # "Multi-task training beneficial: similarity 0.73 > threshold 0.5"

# 5. Build the joint model
model: MultiTaskModel = strategy.execute_transfer(source_task, target_task, backbone)

# 6a. Train with uncertainty weighting (self-calibrating)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

x = torch.randn(16, 25)
batch = {
    str(source_task.task_id): (x, torch.randint(0, 3, (16,))),
    str(target_task.task_id): (x, torch.randint(0, 5, (16,))),
}

loss = model.compute_uncertainty_loss(batch)
loss.backward()
opt.step()

# 6b. Train with equal weighting
model_equal = MultiTaskTransfer(backbone, task_weighting="equal")
model_equal.add_task(source_task, 3)
model_equal.add_task(target_task, 5)
mt_model = model_equal.execute_transfer(source_task, target_task, backbone)

loss_equal = mt_model.compute_loss(batch, model_equal.task_weights)
loss_equal.backward()
```

#### `orcanet.retrieval` — Hybrid Retrieval

Three-stage async pipeline that narrows from broad vector similarity to a small set of highly relevant candidates, with optional LLM re-ranking. Public exports from `orcanet.retrieval`: `QueryExpander`, `LLMRanker`, `HybridRetriever`.

##### `QueryExpander`

Generates alternative phrasings of a task description to broaden FAISS recall before the vector search stage.

```python
from orcanet.retrieval import QueryExpander

expander = QueryExpander(llm=my_llm)
alternatives = await expander.expand(
    "brain MRI binary classification",
    n_expansions=3,
)
# → ["medical image classification", "neurological imaging task", "3D scan binary classification"]
```

| Method / Helper | Signature | Description |
|---|---|---|
| `__init__` | `(llm: BaseLLM) → None` | Stores the LLM; no model loading occurs at construction time. |
| `expand` | `async (query: str, n_expansions: int = 3) → list[str]` | Builds a prompt asking the LLM to produce `n_expansions` alternative descriptions, calls `ainvoke`, and parses the result via `_parse_list_from_response`. |
| `_parse_list_from_response` | `(text: str) → list[str]` | Strips numbered prefixes (`1.`, `2)`), bullet markers (`-`, `*`, `•`), and leading/trailing whitespace from each line, then filters blank lines. Returns all non-empty strings. Works on any mix of list formats in a single response. |

##### `LLMRanker`

Re-ranks a list of candidate tasks against a query task by asking an LLM to score each candidate on a 0–1 relevance scale. Output is Pydantic-validated before returning.

```python
from orcanet.retrieval import LLMRanker

ranker = LLMRanker(llm=my_llm)
ranked = await ranker.rerank(query_task, candidates, top_k=5)
# → [(task_a, 0.92, "same domain and similar n_classes"), ...]
```

**Internal Pydantic models:**

```python
class _RankedItem(BaseModel):
    task_id: str
    score: float = Field(ge=0.0, le=1.0)   # strictly validated; rejects scores outside [0, 1]
    reasoning: str

class _RankedList(BaseModel):
    rankings: list[_RankedItem]
```

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(llm: BaseLLM) → None` | Stores the LLM. |
| `rerank` | `async (query_task: Task, candidate_tasks: list[Task], top_k: int = 10) → list[tuple[Task, float, str]]` | Returns `[]` immediately when `candidate_tasks` is empty (no LLM call). Otherwise builds a prompt from query and candidate metadata, calls `ainvoke`, strips markdown backtick fences, and delegates to `_parse_ranked_list`. Results are sorted by score descending and truncated to `top_k`. |
| `_parse_ranked_list` | `(text: str, candidate_tasks: list[Task]) → list[tuple[Task, float, str]]` | Strips leading ` ```json ` / ` ``` ` fences, then validates via `_RankedList.model_validate_json`. Any `ValidationError`, `json.JSONDecodeError`, or `ValueError` returns `[]` with a `WARNING`-level log. Task IDs in the LLM output that are absent from `candidate_tasks` are silently skipped. |

The prompt template (`_RERANK_PROMPT_TEMPLATE`) instructs the LLM to return **only** a JSON object with no markdown or explanation, lists all `n_candidates` candidates with their metadata, and names the exact field types and constraints for `score`.

##### `HybridRetriever`

Three-stage retrieval pipeline. All dependencies are constructor-injected for testability.

```python
from orcanet.retrieval import HybridRetriever

retriever = HybridRetriever(
    faiss_index=index,
    task_repository=repo,
    embedder=cross_domain_embedder,
    query_expander=expander,
    llm_ranker=ranker,
    top_k_initial=50,
    top_k_final=10,
    similarity_threshold=0.6,
    use_llm_reranking=True,
)
results = await retriever.retrieve(query_task, filters={"domain": "vision"})
# → [(task, score, reasoning), ...]
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `faiss_index` | object with `.search(embedding, k) → list[tuple[str, float]]` | — | FAISS index returning `(task_id_str, cosine_score)` pairs |
| `task_repository` | `TaskRepository` | — | Async repository; `get_by_id(UUID) → Task \| None` is the only method called |
| `embedder` | `CrossDomainEmbedder` | — | Converts the 25-dim feature vector to the 64-dim FAISS embedding space |
| `query_expander` | `QueryExpander` | — | Generates alternative query descriptions for `retrieve_with_expanded_queries` |
| `llm_ranker` | `LLMRanker` | — | Re-ranks the Stage 2 candidate list when Stage 3 is activated |
| `top_k_initial` | `int` | `50` | Number of candidates retrieved from FAISS in Stage 1 |
| `top_k_final` | `int` | `10` | Maximum number of results returned |
| `similarity_threshold` | `float` | `0.6` | FAISS scores below this are dropped in Stage 2 |
| `use_llm_reranking` | `bool` | `True` | When `False`, Stage 3 is skipped regardless of candidate count |

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `retrieve` | `async (query_task: Task, filters: dict \| None = None) → list[tuple[Task, float, str]]` | Executes the three-stage pipeline described below. |
| `retrieve_with_expanded_queries` | `async (query_description: str, query_task: Task) → list[tuple[Task, float, str]]` | Calls `QueryExpander.expand(query_description)`, then for the original description and each expansion creates a task variant via `query_task.model_copy(update={"name": description})` and calls `retrieve(query_variant)`. Merges all results via `_deduplicate_and_sort` and returns the top-`top_k_final` deduplicated entries. Each expansion drives a distinct FAISS embedding so the fan-out is semantically meaningful. |

**Three-stage pipeline (inside `retrieve`):**

1. **Stage 1 — FAISS vector search**: `_task_to_feature_vector(query_task)` produces a 25-dim float32 array — `log1p(n_samples)` at index 0, raw `n_features` at index 1, raw `n_classes` at index 2; `None` fields map to 0. The array is wrapped in a `torch.Tensor`, passed through `CrossDomainEmbedder.embed()` to obtain a unit-normalised 64-dim embedding, and searched against the FAISS index with `k=top_k_initial`. Returns `[]` immediately when the index returns no candidates.

2. **Stage 2 — Metadata filter + threshold**: all candidate IDs are batch-fetched via `asyncio.gather(..., return_exceptions=True)`. Failed fetches (exceptions) are logged at `WARNING` and skipped rather than aborting the entire batch, so a single flaky database row does not discard all other candidates. `None` results (deleted or unknown tasks) are silently dropped. Any candidate whose FAISS score falls below `similarity_threshold` is discarded. The optional `filters` dict then applies field-equality checks: `getattr(task, key) == val` for each `(key, val)` pair.

3. **Stage 3 — LLM re-ranking (optional)**: activated only when `use_llm_reranking=True` **and** the number of surviving candidates exceeds `top_k_final`. Delegates to `LLMRanker.rerank(query_task, candidates, top_k=top_k_final)`. When the condition is not met, returns the top-`top_k_final` candidates each annotated with `"vector similarity"` as the reasoning string.

**Helper functions (module-level):**

- `_task_to_feature_vector(task: Task) → np.ndarray` — 25-dim float32 feature vector. Only the first three indices are currently populated; indices 3–24 are reserved for future statistical features and are always zero.
- `_deduplicate_and_sort(results: list[tuple[Task, float, str]]) → list[tuple[Task, float, str]]` — Collapses entries that share the same `task_id` UUID by keeping the one with the highest score, then sorts the remaining entries descending by score. Used by `retrieve_with_expanded_queries` to merge and rank results from multiple query expansions without duplicates.

#### `orcanet.reasoning` — LangChain Reasoning Agent

LLM-powered reasoning layer that generates structured transfer recommendations over a tool-augmented agent loop. Public exports from `orcanet.reasoning`: `OrcaNetAgent`, `TransferRecommendationResponse`, `SourceTaskRecommendation`, `LLMParsingError`.

##### Response Validators (`orcanet.reasoning.validators`)

Pydantic v2 schemas that type and bound the agent's structured output.

**`LLMParsingError(Exception)`**

Raised by `OrcaNetAgent.recommend_transfer` after all retry attempts are exhausted and the LLM output cannot be parsed into a `TransferRecommendationResponse`.

**`SourceTaskRecommendation(BaseModel)`**

| Field | Type | Constraints | Description |
|---|---|---|---|
| `task_id` | `str` | — | UUID string of the recommended source task |
| `task_name` | `str` | — | Human-readable source task name |
| `similarity_score` | `float` | `[0.0, 1.0]` | Embedding cosine similarity to the target task |
| `transfer_score` | `float` | `[0.0, 1.0]` | Transfer strategy score from `transfer_scoring_tool` |
| `reasoning` | `str` | — | Agent's natural-language rationale for this recommendation |

**`TransferRecommendationResponse(BaseModel)`**

| Field | Type | Constraints | Description |
|---|---|---|---|
| `top_sources` | `list[SourceTaskRecommendation]` | may be empty | Ranked list of recommended source tasks |
| `recommended_strategy` | `TransferStrategy` | `Literal["feature","weight","architecture","multi_task"]` | Transfer strategy name; Pydantic rejects any value not in the allowed set at parse time |
| `expected_improvement` | `float` | `[0.0, 1.0]` | Predicted relative improvement from applying the recommended transfer |
| `explanation` | `str` | — | Comprehensive free-text explanation of the recommendation |
| `confidence` | `float` | `[0.0, 1.0]` | Agent's confidence in the recommendation |

##### LangChain Tools (`orcanet.reasoning.tools`)

Four `@tool`-decorated async functions exposed to the agent loop. Each tool module uses two DI mechanisms:

- **Module-level registry**: a plain `None`-initialised variable with `set_*(instance)` / `get_*()` helpers. The `@tool`-decorated function reads from these globals, providing backwards-compatible standalone use.
- **Per-instance factory** (`make_<tool_name>(*deps) → StructuredTool`): returns a new `StructuredTool.from_function()` instance whose inner coroutine closes over the supplied dependencies. `OrcaNetAgent.__init__` uses these factories so each agent instance owns its own tool objects and **never mutates shared module-level state**. Constructing two agents with different dependencies is therefore safe.

All four tools return JSON-encoded strings and handle errors gracefully by returning `{"error": "..."}` rather than raising, allowing the agent to relay error context to the LLM and attempt a corrective action.

**`task_retrieval_tool(query: str, filters: str = "{}") → str`**

*"Retrieve similar ML tasks from the registry. Returns JSON list of tasks with similarity scores."*

Calls `HybridRetriever.retrieve_with_expanded_queries(query, stub_task)` to perform three-stage hybrid retrieval. Applies any `filters` JSON as post-retrieval field-equality checks on each returned `Task`. Returns a JSON array of objects with `task_id`, `task_name`, `domain`, `task_type`, `n_samples`, `n_classes`, `score`, and `reason` fields. Module-level DI: `set_retriever(r)` / `get_retriever()`.

**`embedding_similarity_tool(task_id_a: str, task_id_b: str) → str`**

*"Compute embedding similarity between two tasks. Returns a float 0-1."*

Fetches both tasks from the repository, builds 25-dim feature vectors, and embeds them via `CrossDomainEmbedder.embed()`. Both embeddings are **explicitly L2-normalised** with `torch.nn.functional.normalize(emb, dim=0)` before the dot product, ensuring a true cosine similarity regardless of whether the embedder returns unit vectors. Returns `{"similarity": float}` clamped to `[-1.0, 1.0]`. Module-level DI: `set_embedder(e)` / `get_embedder()`, `set_task_repository(r)` / `get_task_repository()`. Per-instance factory: `make_embedding_similarity_tool(embedder, task_repository)`.

**`transfer_scoring_tool(source_task_id: str, target_task_id: str, strategy: str = "feature") → str`**

*"Score the transferability between two tasks using the specified strategy."*

Dispatches to a named `TransferStrategy` instance from the registered strategy dict. Returns `{"overall": float, "layer_scores": dict, "recommended_layers": list, "reasoning": str}` reflecting the `TransferScore` returned by `strategy.score_transfer(source, target)`. Returns `{"error": "..."}` if the strategy name is not registered. Module-level DI: `set_transfer_strategies(d: dict)` / `get_transfer_strategies()`, `set_task_repository(r)` / `get_task_repository()`.

**`performance_prediction_tool(task_id: str, model_config_json: str) → str`**

*"Predict the performance of a model configuration on a given task."*

Fetches the task, deserialises `model_config_json`, and calls `OrcaMindClient.predict_performance(task, model_config)`. Returns `{"metrics": {**final_metrics}, "experiment_id": str}`. After parsing, validates `isinstance(model_config, dict)` and returns `{"error": "model_config_json must be a JSON object"}` immediately when the check fails — preventing `AttributeError` from a non-dict JSON value (arrays, strings, numbers). Module-level DI: `set_orcamind_client(c)` / `get_orcamind_client()`, `set_task_repository(r)` / `get_task_repository()`. Per-instance factory: `make_performance_prediction_tool(orcamind_client, task_repository)`.

##### `OrcaNetAgent`

LangChain reasoning agent that runs a tool-augmented loop and returns a validated `TransferRecommendationResponse`. Uses `langchain.agents.create_agent` (LangChain 1.x langgraph `CompiledStateGraph` API) with a locally-defined system prompt, eliminating any network access to `langchain hub`.

```python
from orcanet.reasoning import OrcaNetAgent

agent = OrcaNetAgent(
    llm_provider="openai",          # "openai" | "anthropic" | "local"
    model="gpt-4-turbo",
    temperature=0.7,
    api_key="sk-...",
    retriever=hybrid_retriever,
    embedder=cross_domain_embedder,
    task_repository=task_repo,
    transfer_strategies={"feature": feature_strategy, "weight": weight_strategy},
    orcamind_client=orcamind_client,
)

result = await agent.recommend_transfer("find the best source task for retinal scan classification")
print(result.recommended_strategy)     # "feature"
print(result.top_sources[0].task_name) # "brain MRI classification"
print(result.confidence)               # 0.82
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `llm_provider` | `str` | `"openai"` | LLM backend: `"openai"` → `ChatOpenAI`; `"anthropic"` → `ChatAnthropic`; `"local"` → `ChatOpenAI` with base URL from `ORCANET_LOCAL_LLM_URL` env var (default `http://localhost:11434/v1`) |
| `model` | `str` | `"gpt-4-turbo"` | Model name passed to the LLM constructor |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `api_key` | `str \| None` | `None` | API key for the LLM provider |
| `retriever` | `HybridRetriever \| None` | `None` | Injected into `task_retrieval_tool` module registry |
| `embedder` | `CrossDomainEmbedder \| None` | `None` | Injected into `embedding_similarity_tool` module registry |
| `task_repository` | `TaskRepository \| None` | `None` | Injected into `embedding_similarity_tool`, `transfer_scoring_tool`, and `performance_prediction_tool` module registries |
| `transfer_strategies` | `dict \| None` | `None` | Dict mapping strategy name string to `TransferStrategy` instance; injected into `transfer_scoring_tool` |
| `orcamind_client` | `OrcaMindClient \| None` | `None` | Injected into `performance_prediction_tool` module registry |

**`recommend_transfer(query: str) → TransferRecommendationResponse`** (async)

Invokes the agent loop by passing `HumanMessage(content=query)` to the underlying `CompiledStateGraph`. Extracts the final message's `content` string, strips markdown fences if present, and attempts to validate it as a `TransferRecommendationResponse` via `model_validate_json`. Retry logic retries up to `_MAX_RETRIES = 2` additional attempts (3 total). On each failed parse, a corrective `HumanMessage` is constructed as `query + _CORRECTIVE_SUFFIX`, where the suffix instructs the LLM to respond only with valid JSON matching the required schema. After the third failure, `LLMParsingError` is raised.

| Attempt | Parse result | Action |
|---|---|---|
| 1 | Success | Return `TransferRecommendationResponse` immediately |
| 1 | Failure | Log warning; rebuild messages with corrective prompt |
| 2 | Success | Return immediately |
| 2 | Failure | Log warning; rebuild messages with corrective prompt |
| 3 | Success | Return immediately |
| 3 | Failure | Raise `LLMParsingError` |

**`_build_llm(provider, model, temperature, api_key)`**

| `provider` | LLM class | Notes |
|---|---|---|
| `"openai"` | `langchain_openai.ChatOpenAI` | — |
| `"anthropic"` | `langchain_anthropic.ChatAnthropic` | — |
| `"local"` | `langchain_openai.ChatOpenAI` | `base_url` from `ORCANET_LOCAL_LLM_URL` env var; `api_key` defaults to `"local"` |
| any other value | — | Raises `ValueError` naming the invalid provider and listing the three supported values. Fails fast rather than silently defaulting. |

##### Prompt Templates (`orcanet.reasoning.prompts`)

Three string constants exported from `orcanet.reasoning.prompts`:

| Constant | Module | Purpose |
|---|---|---|
| `TRANSFER_EXPLANATION_TEMPLATE` | `transfer_explanation` | Explains why a source task's model transfers to the target task, citing dataset statistics and performance history |
| `TASK_SIMILARITY_TEMPLATE` | `task_similarity` | Assesses structural similarity between two tasks based on domain, feature space, and class distribution |
| `ARCHITECTURE_RECOMMENDATION_TEMPLATE` | `architecture_recommendation` | Recommends an architecture for the target task given similar-task performance history and computational constraints |

#### `orcanet.integration` — Three-Way Pipeline

The `integration/` module coordinates OrcaNet, OrcaMind, and OrcaLab into a single end-to-end transfer-validation workflow. All public names are importable from `orcanet.integration`.

```python
from orcanet.integration import ServiceUnavailableError, TransferPipeline, TransferValidationResult
```

##### `TransferValidationResult` (Pydantic model)

Returned by `TransferPipeline.recommend_and_validate`. All fields are set even on partial success (e.g. when OrcaLab times out, `experiment_result` is `None` but `mapping` is always populated).

| Field | Type | Description |
|---|---|---|
| `score` | `TransferScore` | CKA-based transfer score produced by the chosen strategy |
| `experiment_result` | `ExperimentResult \| None` | OrcaLab experiment result, or `None` when validation was skipped or timed out |
| `mapping` | `TransferMapping` | Persisted `TransferMapping` row written to the database regardless of experiment outcome |
| `improvement_over_baseline` | `float \| None` | `accuracy − baseline_accuracy` from `experiment_result.metrics`; `None` when either metric is absent |

##### `TransferPipeline`

Orchestrates the four-step transfer-validation workflow.

```python
pipeline = TransferPipeline(
    orcamind_client=orcamind_client,
    orcalab_client=orcalab_client,
    transfer_strategies={"feature": FeatureTransfer(...)},
    task_repository=task_repo,
)
result = await pipeline.recommend_and_validate(
    source_task_id="uuid-str",
    target_task_id="uuid-str",
    strategy_name="feature",
    validate=True,
)
```

**`recommend_and_validate` execution steps:**

1. **Resolve tasks** — calls `task_repository.get_by_id()` for both IDs; raises `ValueError` if either is absent.
2. **Get best model** — calls `orcamind_client.get_best_model(source_task_id)`; wraps `httpx.ConnectError`, `httpx.TimeoutException`, and 5xx `HTTPStatusError` in `ServiceUnavailableError`.
3. **Score transfer** — calls `transfer_strategies[strategy_name].score_transfer(source_task, target_task)` using the selected strategy; raises `KeyError` for an unknown name.
4. **Optionally validate** — when `validate=True` and `score.overall > 0.4`, calls `orcalab_client.create_experiment(...)` tagged `["transfer_validation", "source:<id>", "strategy:<name>"]`, then `orcalab_client.wait_for_completion(experiment_id, timeout=3600)`. A `TimeoutError` from `wait_for_completion` is caught, logged, and sets `experiment_result = None` so the pipeline always completes.
5. **Persist mapping** — calls `task_repository.save_transfer_mapping(...)` with the score, strategy name, and experiment metadata; the mapping is written regardless of whether validation ran or timed out.

##### `ServiceUnavailableError`

Raised when OrcaMind is unreachable or returns a server error. The `/api/v1/transfer/validate` endpoint maps this to **HTTP 503**.

#### `orcanet.api` — FastAPI Service (port 8002)

Eight live endpoints served by FastAPI, documented at `GET /docs`. The service runs on port 8002 by default and is launched via `orcanet serve` or `uvicorn orcanet.api.main:app`. See [API Reference](API-REFERENCE.md#orcanet-api----port-8002) for full request/response schema details.

| Method | Path | Status | Description |
|---|---|---|---|
| `GET` | `/` | 200 | Service info `{name, version, status}` |
| `GET` | `/health` | 200 | Parallel health probe — `{status, orcamind, orcalab, llm}`; `llm` is `null` unless `?deep=true`; always 200 even when degraded |
| `POST` | `/api/v1/transfer/score` | 200 / 400 / 404 | CKA-based transfer score between two registered tasks; 400 for unknown strategy, 404 for missing task |
| `POST` | `/api/v1/transfer/recommend` | 200 / 502 | `OrcaNetAgent` recommendation; supports `X-LLM-Provider` header override; 502 on agent error |
| `POST` | `/api/v1/transfer/validate` | 200 / 400 / 404 / 503 | Three-way pipeline: score → optional OrcaLab experiment → persist `TransferMapping`; 503 when OrcaMind is unreachable, 400 for unknown strategy, 404 for missing task |
| `GET` | `/api/v1/transfer/{mapping_id}` | 200 / 404 / 422 | Stored `TransferMapping` record from DB; 404 when not found |
| `POST` | `/api/v1/retrieve` | 200 / 404 | Three-stage hybrid retrieval; LLM query expansion when `query_description` is provided |
| `POST` | `/api/v1/explain` | 200 / 502 | LLM-generated explanation for a source→target transfer; supports `X-LLM-Provider` header; 502 on parse failure |
| `POST` | `/api/v1/cross-domain-embed` | 200 / 404 / 422 | 64-dim L2-normalised embedding from `CrossDomainEmbedder`; accepts `task_id` or `statistical_features` |

**Implementation structure:**

| File | Role |
|---|---|
| `api/main.py` | `create_app()` factory; ASGI lifespan initialises DB engine, `CrossDomainEmbedder`, FAISS index, `HybridRetriever`, `OrcaNetAgent`, `OrcaMindClient`, `OrcaLabClient`, and transfer strategies dict |
| `api/deps.py` | `Depends()` providers for all services; `get_orcanet_agent()` validates the `X-LLM-Provider` header against the allowed set and injects a request-scoped `TaskRepository` into override agents; `get_transfer_pipeline()` constructs a `TransferPipeline` wired to `app.state` clients |
| `api/schemas.py` | `TransferScoreRequest`, `TransferRecommendRequest`, `TransferValidateRequest` (`run_validation` field aliased as `validate`), `RetrieveRequest`, `EmbedRequest` (XOR validation + 25-dim pin on `statistical_features`), `ExplainRequest`, `EmbedResponse`, `ExplainResponse` |
| `api/middleware.py` | CORS (`CORS_ORIGINS` env) + request-logging `BaseHTTPMiddleware` |
| `api/routers/transfer.py` | Score, recommend, validate, and mapping lookup endpoints |
| `integration/pipeline.py` | `TransferPipeline` orchestrator; `TransferValidationResult` schema; `ServiceUnavailableError` |
| `api/routers/retrieve.py` | Retrieve endpoint; delegates to `HybridRetriever` |
| `api/routers/explain.py` | Explain endpoint; catches `LLMParsingError` → 502 |
| `api/routers/embed.py` | Cross-domain embed endpoint; accepts task ID or raw feature vector |

**`X-LLM-Provider` header override** — any request to `/transfer/recommend` or `/explain` may include this header set to `openai`, `anthropic`, or `local`. Values outside this closed set return **400** immediately. When a valid provider is present, `get_orcanet_agent()` constructs a fresh `OrcaNetAgent` using that provider, the `ORCANET_LLM_API_KEY` environment variable, and a request-scoped `TaskRepository` — so override agents retain full task-lookup and transfer-scoring capabilities, identical to the shared singleton.

**Health check internals** — `GET /health` fires two concurrent tasks via `asyncio.gather`:
1. `httpx.AsyncClient.get(orcamind_url + "/health")` with 3 s timeout
2. `httpx.AsyncClient.get(orcalab_url + "/health")` with 3 s timeout

The LLM check (`agent.llm.ainvoke("ping")`, 5 s timeout via `asyncio.wait_for`) is opt-in and only runs when the `?deep=true` query parameter is supplied. Omitting it avoids recurring token costs from frequent load-balancer probes. The `llm` field in the response is `null` for shallow probes. Any exception sets the corresponding flag to `false`. The endpoint never returns a non-200 status.

**Lifespan shutdown** — the three cleanup tasks (`engine.dispose()`, `orcamind_client.aclose()`, `orcalab_client.aclose()`) run concurrently via `asyncio.gather(return_exceptions=True)` so a failure in any one does not prevent the others from completing.

**Integration test suite** — `tests/integration/api/` uses `httpx.AsyncClient` + `ASGITransport` (no lifespan). All services are pre-populated via `app.dependency_overrides` and manual `app.state` assignment. OrcaMind/OrcaLab HTTP calls in the health tests are intercepted by `respx`. The async `client` fixture is decorated with `@pytest_asyncio.fixture` for compatibility with pytest-asyncio strict mode. Coverage of the new API code exceeds 90 %.

### CLI (`orcanet`)

Two commands installed as the `orcanet` entry point.

```bash
orcanet --help           # List all commands
orcanet <command> --help # Per-command usage
```

| Command   | Purpose                                      | Key Options                               |
|-----------|----------------------------------------------|-------------------------------------------|
| `serve`   | Start the FastAPI server on port 8002        | `--host TEXT`, `--port INT`, `--reload`   |
| `version` | Print the OrcaNet package version            | —                                         |

The `serve` command binds `0.0.0.0` inside the container (network namespace isolation). The `host` binding carries an inline `# noqa: S104` suppression with an explanatory comment.

### Hydra Configuration (`config/`)

```text
config/
├── config.yaml              # Root: llm (provider, model, temperature), retrieval thresholds, orcamind/orcalab URLs
├── retriever/
│   └── hybrid.yaml          # FAISS index path, database_url (required from env), top_k thresholds
├── embedder/
│   └── cross_domain.yaml    # DANN dims: input=25, embedding=64, n_domains=10, n_task_types=3
└── llm/
    └── openai.yaml          # Provider (openai), model (gpt-4-turbo), temperature, max_tokens=2048
```

All sensitive values use `${oc.env:VAR}` (required) or `${oc.env:VAR,default}` (optional with fallback). The `database_url` in `retriever/hybrid.yaml` is declared required (`${oc.env:DATABASE_URL}` with no fallback) to prevent the retriever from silently connecting to a wrong database.

| Config key | Resolution |
|---|---|
| `llm.api_key` | `${oc.env:OPENAI_API_KEY,}` — empty string when absent; LLM re-ranking step skipped |
| `retrieval.database_url` | `${oc.env:DATABASE_URL}` — required; no fallback |
| `orcamind.api_url` | `${oc.env:ORCAMIND_API_URL,http://localhost:8000}` |
| `orcalab.api_url` | `${oc.env:ORCALAB_API_URL,http://localhost:8001}` |
| `embedder.model_path` | `${oc.env:CROSS_DOMAIN_MODEL_PATH,/data/models/cross_domain_embedder.pt}` |

### Test Infrastructure (`tests/`)

Fourteen unit test files across five directories, plus two empty `__init__.py` placeholders for future integration tests.

| File | Tests | Covers |
|---|---|---|
| `tests/unit/test_package.py` | 4 | Package importability, `__version__` string, parametrized submodule imports (6 submodules in one test), no unexpected `sys.modules` side effects on `import orcanet.embeddings` (enforces the lazy-import shim) |
| `tests/unit/test_cli.py` | 4 | `version` output matches `__version__`, `--help` exit 0, `serve --help` shows `--host`/`--port`/`--reload`, `no_args_is_help=True` shows both commands |
| `tests/unit/test_config.py` | 10 | Root config key presence, retrieval defaults, LLM defaults, hybrid retriever YAML, cross-domain embedder dimensions, OpenAI LLM YAML, all `.yaml` files parseable by OmegaConf |
| `tests/unit/embeddings/test_cross_domain.py` | 15 | `GradientReversalLayer` gradient negation, alpha scaling, forward identity, zero-alpha passthrough; `CrossDomainEmbedder` output shape, L2 normalisation, eval/training mode preservation across `embed()` calls; DANN task-loss convergence over 20 epochs; domain invariance geometric dispersion (within- vs. cross-domain cosine distance std ratio); `save`/`load` roundtrip embedding equality; config attribute round-trip fidelity |
| `tests/unit/embeddings/test_text_features.py` | 16 | `TextTaskEmbedder` shape and L2-normalisation for `embed_from_description`; invalid fusion `ValueError`; `embed_with_stats` output shape for all three fusion strategies and the no-labels regression path; output L2-normalisation for fused vectors; custom `output_dim` via the `add` fusion; semantic similarity ordering (image tasks cluster closer together than they do to financial tasks); identical descriptions produce dot-product 1.0; batch shape `(N, 384)`, row-wise normalisation, and exact numerical match against individual `embed_from_description` calls |
| `tests/unit/embeddings/test_architecture_embedder.py` | 45 | `ArchitectureGraph` node counts for MLP, CNN, single-layer, and empty configs; node feature shape `(n, 16)` and dtype float32; one-hot correctness for all layer types and all activation types; log-size encoding against `_LOG_SIZE_SCALE`; unknown-type zero-out; sequential and skip-connection edge counts; edge dtype int64; global features shape, dtype, depth equality, log-total-size sign, log-max-width value; `ArchitectureEmbedder.embed` shape `(32,)`, L2 norm, numpy dtype float32, custom `output_dim`, training-mode preservation (both directions), single-layer and empty-config edge cases, determinism across two calls; similarity — identical configs → 1.0, self-similarity exceeds cross-architecture similarity (seeded), return type float, symmetry; retrieval — `top_k` count, fewer-candidates-than-k, descending sort, tuple types, identical query is top result, default `top_k=5`, `top_k=0` returns `[]`, negative `top_k` raises `ValueError` |
| `tests/unit/transfer/test_feature_transfer.py` | 37 | `linear_cka` — self-similarity equals 1.0 (square, tall, wide); orthogonal subspace pair near zero; CKA symmetry; value always in [0, 1]; different feature dimensions allowed; return type float. `FeatureTransfer` — identical models produce overall > 0.9; all layer scores near 1.0; all layers recommended; `TransferScore` instance returned; random-init models score well below identical; reasoning non-empty; layer scores populated. Guards — `ValueError` when source/target not registered; `ValueError` when `probe_data` absent; mock `OrcaMindClient` stored but not called during scoring. Metadata — strategy name, threshold, probe-data presence flag, registered model count. Structural invariants — overall in [0, 1]; `layer_scores` is dict; `recommended_layers` is list; reasoning non-empty string; recommended layers are a subset of layer_scores keys; recommended layers all exceed threshold. `execute_transfer` — returns `nn.Module`; does not mutate source model; result differs from unmodified target; `ValueError` when target not registered |
| `tests/unit/transfer/test_weight_transfer.py` | 34 | `WeightTransfer` score — identical models produce `overall == 1.0`; all layer scores are 1.0; all params in `recommended_layers`. Match-by semantics — `"name"` mode scores 1.0 for all params even when last-layer shapes differ; `"both"` mode excludes shape-mismatched params and drops `overall` below 1.0; first-layer params remain 1.0 in `"both"` mode; `"shape"` mode score is a float in [0, 1]. Structural invariants — `overall` in [0, 1]; `layer_scores` is dict; `recommended_layers` is list and subset of `layer_scores` keys; all recommended layers have score 1.0; reasoning non-empty string. `execute_transfer` — returns `(nn.Module, list[str])` tuple; transferred-layer weights equal source weights after transfer; all params transferred for identical architecture; source model not mutated; no exception for any `match_by` mode; `_safe_reinit` handles 1-D bias and 2-D+ weight tensors without raising. `get_optimizer_with_layer_lr` — returns `torch.optim.Adam`; transferred params get `base_lr * decay`; non-transferred params get `base_lr`; `decay=0.0` produces zero LR for transferred layers; empty transferred list gives all-base-lr groups. Guards — `ValueError` when source/target not registered; `ValueError` on invalid `match_by`. Metadata — `strategy == "weight_transfer"`; `match_by`, `frozen_epochs`, `layer_lr_decay` reflected; defaults verified. |
| `tests/unit/retrieval/test_query_expander.py` | 12 | `_parse_list_from_response` — numbered lists (`1.`, `2.`), dash bullets, asterisk bullets, blank-line filtering, plain-line passthrough, empty-string input returns `[]`, every returned element is non-empty. `QueryExpander.expand` — correct expansion count, each expansion non-empty, prompt contains `n_expansions` integer and `query` text, empty LLM response returns `[]`, default `n_expansions=3` reflected in the prompt. |
| `tests/unit/retrieval/test_ranker.py` | 11 | `_parse_ranked_list` — valid JSON parsed into `(Task, float, str)` tuples with correct field values; unknown `task_id` in LLM output excluded; markdown ` ```json ` / ` ``` ` fences stripped before parsing; invalid JSON returns `[]`; `score` outside `[0, 1]` triggers `ValidationError` and returns `[]`; three-candidate input returns all three tuples. `LLMRanker.rerank` — empty `candidate_tasks` returns `[]` without calling `ainvoke`; results sorted descending by score; `top_k` truncates output; prompt contains query task `name`, `domain`, and `task_type`; prompt contains all candidate `task_id` strings. |
| `tests/unit/retrieval/test_retriever.py` | 12 | `_task_to_feature_vector` — shape `(25,)`, dtype float32; `n_samples=1000` maps to `log1p(1000)` at index 0; all-`None` statistical fields produce an all-zero vector. `_deduplicate_and_sort` — duplicate `task_id` entries collapsed to the highest-scoring entry; output sorted descending by score; single-entry input returned unchanged. `HybridRetriever.retrieve` — empty FAISS result returns `[]`; candidate with FAISS score 0.4 excluded when `similarity_threshold=0.6`; `use_llm_reranking=False` leaves `LLMRanker.rerank` uncalled; `LLMRanker.rerank` called exactly once when three candidates exceed `top_k_final=2`; `filters={"domain": "nlp"}` removes task with `domain="vision"`. `retrieve_with_expanded_queries` — same `task_id` returned from multiple expansions appears exactly once in the output. |
| `tests/unit/reasoning/test_validators.py` | 15 | `LLMParsingError` — subclasses `Exception`; carries message; can be raised and caught. `SourceTaskRecommendation` — valid construction with all five fields; `similarity_score < 0` rejected; `transfer_score > 1` rejected; missing required field raises `ValidationError`. `TransferRecommendationResponse` — valid JSON parsed via `model_validate_json`; invalid JSON raises; `confidence > 1` rejected; `expected_improvement < 0` rejected; all four strategy names accepted; empty `top_sources` is valid; multiple sources parsed; boundary values `confidence=0.0` and `expected_improvement=1.0` accepted. |
| `tests/unit/reasoning/test_tools.py` | 17 | `TestToolDocstrings` — each of the four tools has a non-empty `description` (LangChain uses this as the tool description in the agent prompt). `TestTaskRetrievalTool` — returns JSON list with `task_id`, `score`, `reason` fields; returns `{"error": "..."}` when retriever not configured; `filters` JSON applied post-retrieval (domain mismatch returns `[]`); default empty filters pass all results. `TestEmbeddingSimilarityTool` — returns `{"similarity": float}` in `[-1, 1]`; returns error when not configured; returns error when task not found. `TestTransferScoringTool` — `"feature"` strategy returns `{"overall": 0.75, "recommended_layers": [...], ...}`; unknown strategy returns error; returns error when not configured. `TestPerformancePredictionTool` — returns `{"metrics": {"predicted_score": ...}}` from `OrcaMindClient`; returns error when not configured; returns error for missing task. |
| `tests/unit/reasoning/test_agent.py` | 10 | `TestOrcaNetAgentTools` — agent constructed with four tools; all four tool names present (`task_retrieval_tool`, `embedding_similarity_tool`, `transfer_scoring_tool`, `performance_prediction_tool`). `TestRecommendTransfer` — returns `TransferRecommendationResponse` on valid JSON; parses `top_sources` correctly; strips markdown fences before parsing; invalid JSON triggers retry (`ainvoke` called 3× before success); two exhausted retries raise `LLMParsingError` with `ainvoke` called exactly 3×; corrective prompt used on retry (`original query + "JSON"` in second `HumanMessage`). `TestBuildLlm` — `llm_provider="openai"` constructs `ChatOpenAI`; `llm_provider="anthropic"` constructs `ChatAnthropic`. |

**Notable patterns introduced in the OrcaNet test suite:**

- *Parametrized submodule imports* — the six submodule import tests are expressed as a single `@pytest.mark.parametrize("submodule", [...])` test rather than six separate functions. Adding a new namespace requires only a new entry in the parameter list.
- *Per-test CLI runner fixture* — `CliRunner` is instantiated via a `scope="function"` pytest fixture, not at module level, so runner state never leaks between tests.
- *pyproject.toml anchor path resolution* — `test_config.py` locates `config/` by walking ancestor directories until a `pyproject.toml` is found. This is robust to test file moves and avoids the fragile `parents[N]` depth index that breaks when the file is relocated.
- *Training-mode preservation testing* — `test_embed_preserves_training_mode` and `test_embed_preserves_eval_mode` verify that `embed()` does not permanently mutate the model's training state, asserting correctness in both directions (model in `.train()` stays in training mode after `embed()` returns; model already in `.eval()` stays in eval mode). This matters because `embed()` is commonly called from inside a training loop for online evaluation.
- *Domain invariance as a geometric assertion* — `TestDomainInvariance.test_within_vs_cross_domain_spread` quantifies the invariance property by computing the standard deviation of cosine distances within each domain and across domains on domain-shifted synthetic data, then asserting the ratio lies in [0.3, 3.0]. This avoids testing exact cluster assignments (which would be brittle) while still enforcing the geometric property that the retrieval layer depends on.
- *Offline SentenceTransformer stub* — `tests/unit/embeddings/conftest.py` patches `orcanet.embeddings.text_features.SentenceTransformer` with `_DeterministicSentenceTransformer`, a session-scoped autouse fixture that never downloads model weights. The stub maps a 36-keyword domain vocabulary (vision, financial, medical, NLP, general ML) to fixed dimensions and fills the remaining 348 dimensions with low-amplitude deterministic noise (σ = 0.05, seeded by `hash(text)`). This design guarantees that semantic ordering tests — e.g. image tasks cluster closer to each other than to financial tasks — hold without a network connection or a local model cache.
- *Relative similarity assertion over fixed thresholds* — `test_similarity_mlp_vs_cnn_less_than_self_similarity` asserts that `embedder.similarity(mlp, mlp) > embedder.similarity(mlp, cnn)` rather than checking `sim < 0.9`. The relative comparison holds for any valid embedder regardless of the random weight initialisation or which message-passing backend is active, avoiding the fragility of a hardcoded cosine cutoff that can shift across seeds or optional dependencies.
- *Boundary tests for public API contracts* — `test_find_similar_top_k_zero_returns_empty` and `test_find_similar_negative_top_k_raises` pin the `find_similar_architectures` boundary behaviour. Without the explicit `ValueError` guard, Python's negative-slice semantics (`scored[:-1]`) would silently return a near-full list instead of raising, making the contract invisible to callers. The boundary tests lock this in so future refactors cannot regress it.
- *Relaxed threshold for shallow-network CKA* — `TestFeatureTransferRandomModels.test_overall_below_identical` asserts `score.overall < 0.8` rather than the intuitively tighter `< 0.5`. Two independently initialised shallow MLPs (10→20→5) produce CKA ≈ 0.60 even with no shared training because the shared input distribution induces a consistent covariance structure. The threshold 0.8 is still a meaningful gap from the ≈1.0 of identical models; it just reflects the minimum distinguishable signal for networks at this scale.
- *execute_transfer mutability assertion* — `test_does_not_mutate_source_model` clones every source parameter before the call and asserts exact tensor equality afterwards. Weight-patching operations that mistakenly modify an in-place buffer on the source would be caught here even if the adapted model looks correct.
- *Orthogonal subspace pair construction* — `TestLinearCKAOrthogonal._orthogonal_pair` draws columns from a QR-factored random square matrix rather than generating two independent random matrices. This guarantees `col(X) ⊥ col(Y)` exactly by construction, so the CKA near-zero assertion is a precise geometric claim rather than a probabilistic one.
- *Deepcopy semantics for WeightTransfer* — `test_weight_transfer.py` explicitly documents that `execute_transfer` starts from `deepcopy(source_model)`, so shape mismatches between source and target cannot arise within `execute_transfer` itself. The `test_shape_mismatch_skipped_without_exception` test verifies the no-raise guarantee across all three `match_by` modes; the shape-safety of `_find_source_tensor` is covered separately via `test_safe_reinit_handles_1d_and_2d_tensors`, which calls `_safe_reinit` directly on 1-D (bias) and 2-D+ (weight) tensors and confirms no exception is raised. This split makes the test intent explicit rather than hiding it behind an architecture-mismatch fixture that would not actually exercise the code path.
- *Binary score assertions for WeightTransfer* — `TestWeightTransferScoreMatchBy` uses two model architectures with identical first layers but different `out_dim` to assert the three `match_by` semantics precisely: `"name"` matches all four parameters (all names exist regardless of shape), `"both"` excludes the two last-layer parameters (shape mismatch), and `"shape"` produces a float in [0, 1] without asserting a specific value (shape-indexed matching on random architectures is deterministic but architecturally coupled). Testing each mode independently makes it easy to add a fourth mode without adjusting existing assertions.
- *Per-parameter optimizer group verification* — `TestGetOptimizerWithLayerLR.test_transferred_params_get_decayed_lr` iterates every `param_groups` entry, resolves the parameter back to its name via `model.named_parameters()`, and asserts the learning rate equals `base_lr * decay` for transferred names and `base_lr` for all others. This catches silent bugs where a group contains the wrong parameter or the LR formula is applied to the wrong set — possible when `param_groups` is built by list comprehension over `named_parameters()` and the index mapping drifts.
- *`SimpleNamespace` mock factory for multi-dependency tests* — `test_retriever.py` uses a `_make_mocks(task)` factory that returns all five `HybridRetriever` dependencies as a `SimpleNamespace`. Tests that need a default happy-path setup call `_make_mocks` and `_build_retriever`; tests that need custom behaviour (e.g. multi-task threshold filtering) replace individual attributes (`mocks.repo.get_by_id = AsyncMock(side_effect=...)`) after construction. This pattern avoids deeply nested fixture pyramids while keeping each test's intent visible in its own body.
- *Real `torch.zeros` tensor in embedder mock* — the FAISS embedding call chain (`CrossDomainEmbedder.embed(tensor).squeeze(0).detach().numpy()`) involves three chained tensor operations. Using `MagicMock().embed.return_value = torch.zeros(25)` rather than a chain of `MagicMock` returns means the chain executes correctly against a real 1-D tensor: `squeeze(0)` is a no-op on a 1-D tensor, `detach()` returns the same tensor, and `numpy()` produces a valid array — so the FAISS mock receives a proper numpy array without needing a spec-heavy mock for the embedder.
- *Autouse reset fixture for shared module state* — `test_tools.py` includes a `_reset_reasoning_tool_state` autouse fixture that calls every `set_*()` helper on all four tool modules with sentinel values (`None`, `{}`) both before and after each test. Module-level registries persist across tests in the same process; without explicit teardown, a test that sets `_retriever` leaks the value into the next test, producing false positives or brittle order dependencies. The before-yield clear guards against leftover state; the after-yield clear restores a clean baseline regardless of whether the test itself called any setters.
- *Scale-invariant cosine similarity assertion* — `test_similarity_is_cosine_regardless_of_embedder_scale` supplies an embedder that returns `torch.ones(1, 64) * 5.0` (un-normalised, constant direction) and asserts `similarity == 1.0`. This is only true if the tool normalises the embeddings before the dot product. The test would pass incorrectly if the tool happened to call an embedder that returned unit vectors, but the constant-scale fixture breaks that coincidence, making the normalisation an explicit, verifiable contract rather than an implicit assumption.
- *`side_effect` lambda for multi-task repository mocking* — when a test needs the repository to return different tasks for different UUIDs (e.g. threshold filtering), `repo.get_by_id = AsyncMock(side_effect=lambda uid: task_map.get(uid))` threads the `UUID` argument through a pre-populated dict. This is preferable to `side_effect=[task1, task2]` (order-dependent) and to separate `AsyncMock` instances (requires separate injection points).
- *Pydantic field-constraint testing via score boundary* — `TestParseRankedList.test_score_out_of_range_returns_empty_list` submits `"score": 1.5` to trigger `_RankedItem`'s `Field(ge=0.0, le=1.0)` constraint, causing a `ValidationError` that `_parse_ranked_list` converts to `[]`. This pins the graceful-degradation contract: a misbehaving LLM that returns an out-of-bounds score never propagates a `ValidationError` to the caller. Testing the boundary directly (not just a valid score) ensures the `Field` constraint is actually present and enforced at parse time rather than being checked post-hoc.
- *`importlib.import_module` to bypass package attribute shadowing* — `tools/__init__.py` exports `task_retrieval_tool` (and the other three) as module-level names, which shadows the submodule of the same name in the package namespace. `import orcanet.reasoning.tools.task_retrieval_tool as mod` resolves via `getattr` on the package, returning the `StructuredTool` object, not the module. `test_tools.py` works around this with a `_mod(name)` helper that calls `importlib.import_module(f"orcanet.reasoning.tools.{name}")`, which looks up `sys.modules` directly and returns the actual module object. Any project that exports a symbol with the same name as a submodule should anticipate this gotcha in tests.
- *`side_effect` list for agent retry sequencing* — `test_invalid_json_triggers_retry` and `test_two_retries_exhausted_raises_llm_parsing_error` use `AsyncMock(side_effect=[response1, response2, response3])` on the `CompiledStateGraph.ainvoke` mock. Each call pops the next element from the list, so the test can exercise the exact sequence of agent responses without any shared state between calls. The `call_count` assertion afterwards locks in the number of retry attempts.
- *`__new__` bypass for agent construction in fixtures* — the `_build_agent` helper uses `OrcaNetAgent.__new__(OrcaNetAgent)` to create an agent instance without running `__init__`, then manually sets `agent.tools`, `agent.llm`, and `agent._agent` to mocks. This avoids the LLM constructor call (which would attempt a real network connection) while still testing the full `recommend_transfer` code path. The pattern is appropriate here because `__init__`'s only side effects are DI registry calls and LLM construction — both tested separately in `TestBuildLlm`.
- *`call_args_list` index access for corrective prompt assertion* — `test_corrective_prompt_used_on_retry` accesses `mock_graph.ainvoke.call_args_list[1][0][0]["messages"]` to inspect the `messages` argument passed on the second `ainvoke` call specifically. This verifies that the corrective prompt is sent on the retry, not on the first call, and that the corrective message contains both the original query string and the word `"JSON"` — confirming the `_CORRECTIVE_SUFFIX` is appended correctly.
