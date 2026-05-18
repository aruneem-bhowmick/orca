# API Reference

> Part of the [Orca](../README.md) meta-learning platform.

---

## Overview

The Orca platform exposes two independent HTTP services:

| Service   | Default Port | Base URL              | Description                             |
|-----------|-------------|----------------------|-----------------------------------------|
| OrcaLab   | `8001`       | `http://localhost:8001` | Experiment orchestration and search      |
| OrcaMind  | `8000`       | `http://localhost:8000` | Meta-learning engine and recommendations |

Both services auto-generate interactive API docs at `GET /docs` (Swagger UI) and `GET /redoc`.

All endpoints use the `/api/v1/` prefix. All request and response bodies are JSON. There is no authentication layer in the current release.

---

## OrcaLab API — port 8001

### Experiments

#### `POST /api/v1/experiments` — Create experiment

Status: **201 Created**

**Request body**

```json
{
  "task_id":        "uuid | null",
  "model_id":       "uuid | null",
  "training_config": {
    "batch_size": 32,
    "lr":         0.001,
    "epochs":     10,
    "optimizer":  "adam",
    "scheduler":  "string | null",
    "extra":      {}
  },
  "created_by": "string | null"
}
```

**Response** — `ExperimentResult`

```json
{
  "experiment_id": "uuid",
  "task_id":       "uuid | null",
  "model_id":      "uuid | null",
  "status":        "pending",
  "mlflow_run_id": "string | null",
  "started_at":    "datetime | null",
  "completed_at":  "datetime | null",
  "metrics":       {}
}
```

---

#### `GET /api/v1/experiments` — List experiments

**Query parameters**

| Parameter | Type  | Default | Constraints    |
|-----------|-------|---------|----------------|
| `limit`   | `int` | `50`    | 1 – 500        |
| `offset`  | `int` | `0`     | ≥ 0            |

**Response** — `list[ExperimentResult]`

---

#### `GET /api/v1/experiments/{experiment_id}` — Get experiment

**Response** — `ExperimentResult` or **404**

---

#### `DELETE /api/v1/experiments/{experiment_id}` — Cancel experiment

Transitions the experiment to `CANCELLED`. Only experiments in `pending`, `queued`, or `running` state can be cancelled.

**Response** — updated `ExperimentResult`, or **404** / **409** (already in a terminal state)

---

#### `WS /api/v1/experiments/{experiment_id}/live` — Live metrics stream

WebSocket endpoint. Polls the database every 2 seconds and pushes a JSON frame to the client.

**Frame payload**

```json
{
  "experiment_id": "uuid-string",
  "status":        "running",
  "epoch":         3,
  "loss":          0.4721,
  "metrics":       { "loss": 0.4721, "epoch": 3 }
}
```

| Field           | Type              | Notes                                                                                     |
|-----------------|-------------------|-------------------------------------------------------------------------------------------|
| `experiment_id` | `string` (UUID)   | Always present                                                                            |
| `status`        | `string`          | One of `pending`, `queued`, `running`, `completed`, `failed`, `cancelled`                 |
| `epoch`         | `int \| null`     | Current epoch number; `null` before the first runner write                                |
| `loss`          | `float \| null`   | Training loss for the most recently completed epoch; `null` before the first runner write |
| `metrics`       | `object`          | Full `experiments.metrics` JSONB dict — backward-compatible envelope of `epoch` + `loss` |

**Error frame** (experiment not found):

```json
{ "error": "experiment not found" }
```

**Connection lifecycle:**
- The server closes the connection automatically when `status` reaches `completed`, `failed`, or `cancelled`.
- `WebSocketDisconnect` from the client is handled gracefully — the server logs the disconnect and exits cleanly.

---

### Sweeps

#### `POST /api/v1/sweeps` — Start sweep

Status: **202 Accepted**

Triggers a Prefect `meta_informed_sweep/default` deployment flow run and registers sweep state in memory. When `PREFECT_API_URL` is not set, no flow is triggered but the sweep record is still created.

**Request body** — `StartSweepRequest`

```json
{
  "task_id":      "string",
  "n_trials":     50,
  "use_orcamind": true,
  "search_space": { } 
}
```

| Field          | Type      | Default | Constraints |
|----------------|-----------|---------|-------------|
| `task_id`      | `string`  | —       | required    |
| `n_trials`     | `int`     | `50`    | ≥ 1         |
| `use_orcamind` | `bool`    | `true`  | —           |
| `search_space` | `object \| null` | `null` | Optuna-compatible search space definition |

**Response**

```json
{ "sweep_id": "uuid-string" }
```

---

#### `GET /api/v1/sweeps/{sweep_id}` — Get sweep status

**Response** — `SweepStatus` or **404**

```json
{
  "sweep_id":      "string",
  "task_id":       "string",
  "n_trials_total": 50,
  "n_completed":   12,
  "n_failed":      1,
  "best_result":   { "trial_id": "...", "objective": 0.91, "params": {}, "status": "completed" },
  "flow_run_id":   "uuid-string | null"
}
```

---

#### `GET /api/v1/sweeps/{sweep_id}/results` — Get sweep trial results

Returns all completed trials sorted by `objective` descending.

**Response** — `list[TrialResult]` or **404**

```json
[
  {
    "trial_id":  "string",
    "objective": 0.91,
    "params":    { "lr": 0.001, "batch_size": 64 },
    "status":    "completed"
  }
]
```

---

### Search Spaces

#### `POST /api/v1/search-spaces` — Create search space

Status: **201 Created**

**Request body** — `CreateSearchSpaceRequest`

```json
{
  "name":        "my_space",
  "description": "optional description",
  "parameters": [
    { "name": "lr", "type": "log_float", "low": 1e-5, "high": 0.1 },
    { "name": "batch_size", "type": "int", "low": 16, "high": 256 }
  ]
}
```

**Response** — `SearchSpaceRecord`

```json
{
  "search_space_id": "uuid",
  "name":            "my_space",
  "definition":      { "name": "...", "description": "...", "parameters": [ ] },
  "parent_id":       null,
  "created_at":      "datetime"
}
```

---

#### `GET /api/v1/search-spaces` — List search spaces

**Query parameters**: `limit` (default 50, max 500), `offset` (default 0)

**Response** — `list[SearchSpaceRecord]`

---

## OrcaMind API — port 8000

### Tasks

#### `GET /api/v1/tasks` — List tasks

**Query parameters**

| Parameter   | Type     | Default | Notes                                              |
|-------------|----------|---------|----------------------------------------------------|
| `domain`    | `string` | —       | Filter by domain; mutually exclusive with task_type |
| `task_type` | `string` | —       | Filter by type; mutually exclusive with domain     |
| `limit`     | `int`    | `50`    | 1 – 500                                            |
| `offset`    | `int`    | `0`     | ≥ 0                                                |

Specifying both `domain` and `task_type` returns **422**.

**Response** — `list[TaskSummary]`

```json
[
  {
    "task_id":   "uuid",
    "name":      "iris_classification",
    "domain":    "classification",
    "task_type": "multiclass"
  }
]
```

---

#### `GET /api/v1/tasks/{task_id}` — Get task

**Response** — `Task` or **404**

```json
{
  "task_id":    "uuid",
  "name":       "string",
  "domain":     "string | null",
  "task_type":  "string",
  "n_samples":  150,
  "n_features": 4,
  "n_classes":  3,
  "dataset_uri": "s3://... | null",
  "metadata":   {},
  "embedding_id": "uuid | null",
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

---

#### `GET /api/v1/tasks/{task_id}/embedding` — Get task embedding

**Response** — `Embedding` or **404** (task not found or task has no embedding)

```json
{
  "embedding_id":     "uuid",
  "task_id":          "uuid | null",
  "embedding_type":   "statistical | neural | null",
  "embedding_vector": [0.12, -0.34, ...],
  "dimension":        25,
  "model_version":    "v1 | null",
  "created_at":       "datetime"
}
```

---

#### `POST /api/v1/tasks/embed` — Embed task

Store a pre-computed embedding vector for a task and link it as the task's current embedding.

**Request body** — `EmbedTaskRequest`

```json
{
  "task_id":          "uuid",
  "embedding_vector": [0.12, -0.34, ...],
  "embedding_type":   "statistical",
  "model_version":    "v1"
}
```

**Response** — `Embedding` or **404** (task not found)

---

### Recommendations

#### `POST /api/v1/recommend-model` — Get model recommendations

Returns top-*k* model recommendations based on task embedding similarity, ranked by `NearestNeighborSelector`.

**Request body** — `RecommendationRequest`

```json
{
  "task_embedding": [0.12, -0.34, ...],
  "domain":         "string | null",
  "task_type":      "string | null",
  "top_k":          5
}
```

**Response** — `list[ModelRecommendation]` or **503** (FAISS index not trained)

```json
[
  {
    "task_id":         "uuid",
    "model_id":        "uuid",
    "architecture":    "resnet18 | null",
    "predicted_score": 0.87,
    "confidence":      0.72,
    "reasoning":       "string | null"
  }
]
```

---

#### `POST /api/v1/predict-performance` — Predict model performance

**Request body**

```json
{
  "task_embedding": [0.12, ...],
  "model_id":       "uuid"
}
```

**Response** or **404** (model not found) / **503** (predictor not trained)

```json
{
  "model_id":        "uuid",
  "predicted_score": 0.84,
  "confidence":      0.68
}
```

---

#### `POST /api/v1/similar-tasks` — Find similar tasks

FAISS cosine-similarity k-NN lookup over all stored task embeddings.

**Request body**

```json
{
  "task_embedding": [0.12, ...],
  "top_k":          5
}
```

**Response** — `list[SimilarityResult]`

```json
[
  { "task_id": "uuid", "score": 0.94, "rank": 1 },
  { "task_id": "uuid", "score": 0.87, "rank": 2 }
]
```

---

### Models

#### `GET /api/v1/models` — List model architectures

**Query parameters**: `limit` (default 100, max 500)

**Response** — `list[ModelConfig]`

```json
[
  {
    "model_id":        "uuid",
    "name":            "resnet18",
    "architecture":    "cnn",
    "config":          { "depth": 18, "width_multiplier": 1.0 },
    "parameter_count": 11689512,
    "flops":           1820000000
  }
]
```

---

### Adaptation

#### `POST /api/v1/adapt` — Start meta-adaptation job

Dispatches an async meta-adaptation background task and returns immediately. The job transitions the created experiment through `running → completed | failed`.

**Request body** — `AdaptRequest`

```json
{
  "task_id":        "uuid",
  "model_id":       "uuid",
  "training_config": { }
}
```

**Response** or **404** (task or model not found)

```json
{ "job_id": "uuid-string" }
```

Poll OrcaLab (`GET http://localhost:8001/api/v1/experiments/{job_id}`) to check completion status. The experiment record is served by OrcaLab on port 8001, not OrcaMind.

---

### Feedback

#### `POST /api/v1/feedback` — Submit experiment feedback

Logs the actual final metric for a completed experiment, closing the OrcaMind meta-learning loop. The metric is persisted via `PerformanceRepository.log_metric` with `is_final=True`.

**Request body** — `FeedbackRequest`

```json
{
  "experiment_id": "uuid",
  "actual_metric": 0.92,
  "metric_name":   "accuracy",
  "params":        { }
}
```

**Response** or **404** (experiment not found)

```json
{ "accepted": true }
```

---

### Performances

#### `GET /api/v1/performances` — List performance summaries

Returns mean metric values grouped by `(task_name, architecture)` — the data source for the Performance Heatmap dashboard page.

**Query parameters**

| Parameter     | Type     | Default      |
|---------------|----------|--------------|
| `metric_name` | `string` | `"accuracy"` |

**Response** — `list[PerformanceSummary]`

```json
[
  {
    "task_name":     "iris_classification",
    "architecture":  "random_forest",
    "mean_accuracy": 0.953
  }
]
```

---

## Health Endpoints

Both services expose a health endpoint with no authentication requirement.

| Service  | Endpoint           | Healthy response                                     |
|----------|--------------------|------------------------------------------------------|
| OrcaMind | `GET /health`      | `{"status": "ok", "faiss": true \| false}`            |
| OrcaLab  | `GET /health`      | `{"status": "ok", "prefect": "http://..." \| null}`   |

The OrcaMind health endpoint reports `faiss: false` when the FAISS index has not been built yet. The service remains fully operational — only `/recommend-model` and `/similar-tasks` return 503 until the index is populated.

---

## Shared Schema Types

All Pydantic models live in `packages/orca-shared/orca_shared/schemas/`.

| Schema file        | Key types                                                 |
|--------------------|-----------------------------------------------------------|
| `task.py`          | `Task`, `TaskSummary`, `TaskCreate`, `DatasetSummary`     |
| `training.py`      | `TrainingConfig`, `ExperimentResult`                      |
| `recommendation.py`| `RecommendationRequest`, `ModelRecommendation`, `FeedbackRequest` |
| `embedding.py`     | `Embedding`, `SimilarityResult`                           |
| `model.py`         | `ModelConfig`, `ModelSummary`                             |
| `metrics.py`       | `MetricPoint`, `PerformanceMetrics`, `PerformanceSummary` |
| `search_space.py`  | `SearchSpaceRecord`                                       |

