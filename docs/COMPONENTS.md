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
| `experiments`       | Training runs                | `task_id`, `model_id`, `training_config` (JSONB), `status`, `mlflow_run_id`               |
| `performances`      | Per-run metrics              | `metric_name`, `metric_value`, `epoch`, `is_final`, `metadata` (JSONB)                    |
| `transfer_mappings` | Task-to-task transfer scores | `source_task_id`, `target_task_id`, `transfer_score`, `transfer_type`                     |
| `search_spaces`     | Hyperparameter definitions   | `name`, `definition` (JSONB), `parent_id` (self-referential tree)                         |


`tasks` ↔ `embeddings` have a circular foreign key handled by `use_alter=True` in the ORM and a deferred `op.create_foreign_key` in the Alembic migration.

### Repository Layer (`registry/repository.py`)

Async repository pattern over all tables:

- `TaskRepository` — `list_all()`, `list_by_domain()`, `list_by_type()`, `get_by_id()`, `create()`, `update_embedding()`
- `ExperimentRepository` — `create()`, `get_by_id()`, `list_by_task()`, `update_status()`, `mark_complete()`
- `PerformanceRepository` — `log_metric()`, `get_final_metrics()`, `get_history()`, `list_all_with_context()`
- `EmbeddingRepository` — `create()`, `get_by_id()`

### Pydantic v2 Schemas (`schemas/`)

20+ validated models across 7 files:


| File                | Models                                                            |
| ------------------- | ----------------------------------------------------------------- |
| `task.py`           | `TaskCreate`, `Task`, `TaskSummary`, `DatasetSummary`             |
| `embedding.py`      | `Embedding`, `SimilarityResult`                                   |
| `model.py`          | `ModelConfig`, `ModelSummary`                                     |
| `recommendation.py` | `RecommendationRequest`, `ModelRecommendation`, `FeedbackRequest` |
| `training.py`       | `TrainingConfig`, `ExperimentResult`                              |
| `transfer.py`       | `TransferMapping`, `TransferScore`, `TransferRecommendation`      |
| `metrics.py`        | `MetricPoint`, `PerformanceMetrics`, `PerformanceSummary`         |


### Storage Backends (`storage/`)

- `StorageBackend` (ABC): `upload()`, `download()`, `delete()`, `exists()`
- `LocalBackend`: filesystem with path-traversal protection
- `MinIOBackend`: S3-compatible object storage via minio-py

### Experiment Tracking (`tracking/`)

- `OrcaTracker`: async context manager for MLflow run lifecycle
- `MetricLogger`: batch `mlflow.log_metrics()` wrapper
- `ArtifactManager`: `upload_model()` / `download_model()` with `weights_only=True`
- `ModelRegistry`: stage-based versioning (Staging → Production → Archived)

### HTTP Clients (`clients/`)

Async `httpx`-based clients for inter-service calls:

- `OrcaMindClient`: `/api/v1/recommend-model`, `/api/v1/predict-performance`, `/api/v1/similar-tasks`
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

12 endpoints served by FastAPI, documented at `GET /docs`.


| Method | Path                          | Description                                                                   |
| ------ | ----------------------------- | ----------------------------------------------------------------------------- |
| `GET`  | `/`                           | Service info (name, version, status)                                          |
| `GET`  | `/health`                     | Liveness probe — `{status, db, faiss, mlflow}` booleans                       |
| `GET`  | `/api/v1/tasks`               | Paginated task list; filter by `domain` or `task_type`                        |
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
│   ├── flows/         # Prefect flows (single experiment, sweep, meta-informed sweep)
│   └── tasks/         # Prefect tasks (prepare_data, train_model, evaluate, log_results)
├── visualization/     # Streamlit dashboard components and pages
├── api/               # FastAPI application and WebSocket endpoint
└── cli.py             # Typer CLI — 4 commands
```

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
