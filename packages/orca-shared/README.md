# orca-shared

> Shared infrastructure layer for the [Orca](../../README.md) meta-learning platform.

---

orca-shared provides the foundational libraries that [OrcaMind](../orcamind/README.md), [OrcaLab](../orcalab/README.md), and [OrcaNet](../orcanet/README.md) all depend on. It owns the database schema, data contracts, storage abstraction, experiment tracking wrappers, and inter-service HTTP clients — ensuring that all three services share a single source of truth for data access and validation.

## What's Inside

```text
orca_shared/
├── registry/     SQLAlchemy ORM models + async repository layer
├── schemas/      Pydantic v2 data contracts (20+ models)
├── storage/      Pluggable storage backends (local filesystem, MinIO)
├── tracking/     MLflow wrappers (run tracking, artifacts, model registry)
└── clients/      Async httpx clients for inter-service communication
```

### Registry

The registry module manages the PostgreSQL meta-learning database through SQLAlchemy 2.0 mapped models and an async repository layer built on `asyncpg`.

**Seven ORM tables** capture the full meta-learning lifecycle:

| Table | Purpose |
|-------|---------|
| `tasks` | ML datasets and tasks with statistical metadata |
| `embeddings` | Task embedding vectors (statistical, neural) |
| `models` | Model architectures and hyperparameter configs |
| `experiments` | Training runs with lifecycle status and per-epoch metrics |
| `performances` | Granular per-metric, per-epoch records |
| `transfer_mappings` | Pairwise source-to-target transfer scores |
| `search_spaces` | Hyperparameter search space definitions (self-referential tree) |

The repository layer (`TaskRepository`, `ExperimentRepository`, `PerformanceRepository`, `EmbeddingRepository`, `SearchSpaceRepository`) provides async CRUD with pagination, domain filtering, and optimistic concurrency control via conditional status updates.

See [Database](../../docs/DATABASE.md) for the full schema reference and migration history.

### Schemas

Over 20 Pydantic v2 models define the data contracts shared across all services:

| File | Models |
|------|--------|
| `task.py` | `TaskCreate`, `Task`, `TaskSummary`, `DatasetSummary` |
| `embedding.py` | `Embedding`, `SimilarityResult` |
| `model.py` | `ModelConfig`, `ModelSummary` |
| `recommendation.py` | `RecommendationRequest`, `ModelRecommendation`, `FeedbackRequest` |
| `training.py` | `TrainingConfig`, `ExperimentResult` |
| `search_space.py` | `SearchSpaceRecord` |
| `transfer.py` | `TransferMapping`, `TransferScore`, `TransferRecommendation` |
| `metric.py` | `MetricPoint`, `PerformanceMetrics`, `PerformanceSummary` |

### Storage

A `StorageBackend` abstract base class with two implementations:

- **`LocalBackend`** — filesystem storage with path-traversal protection.
- **`MinIOBackend`** — S3-compatible object storage via the `minio` library.

### Tracking

MLflow wrappers for experiment lifecycle management:

- **`OrcaTracker`** — async context manager for MLflow run lifetime (params, metrics, artifacts).
- **`MetricLogger`** — batch metric logging.
- **`ArtifactManager`** — model upload/download with `weights_only=True`.
- **`ModelRegistry`** — stage-based model versioning (Staging, Production, Archived).

### Clients

Async `httpx`-based HTTP clients for inter-service calls:

- **`OrcaMindClient`** — task embedding, model recommendation, performance prediction, feedback.
- **`OrcaLabClient`** — experiment creation and status polling.
- **`OrcaNetClient`** — transfer scoring (stub).

All clients call `response.raise_for_status()` so callers receive `httpx.HTTPStatusError` on failure. Upstream services handle these errors with graceful degradation.

See [Components](../../docs/COMPONENTS.md) for detailed method signatures and behaviour.

## Configuration

orca-shared is a library package with no standalone configuration. Its consumers (OrcaMind, OrcaLab, OrcaNet) supply connection URLs and credentials at runtime via environment variables. See [Deployment](../../docs/DEPLOYMENT.md) for the full variable reference.

## Testing

```bash
pytest packages/orca-shared/tests
```

Eight test modules cover the ORM models, repository layer, Pydantic schemas, storage backends, tracking wrappers, and HTTP client mocking.

See [Development](../../docs/DEVELOPMENT.md) for the full testing guide.

## Tech Stack

| Category | Libraries |
|----------|-----------|
| ORM | SQLAlchemy 2.0, asyncpg |
| Validation | Pydantic v2 |
| Storage | minio |
| Tracking | MLflow |
| Caching | Redis |
| HTTP | httpx |

---

[Back to packages](../README.md)
