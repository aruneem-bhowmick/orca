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

- `TaskRepository` — `list_all()`, `list_by_domain()`, `list_by_type()`, `get_by_id()`, `create()`, `update_embedding()`
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
- `OrcaLabClient`: adaptive search calls (stub, planned)
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
