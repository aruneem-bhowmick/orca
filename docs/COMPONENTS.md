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
