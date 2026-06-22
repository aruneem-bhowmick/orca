# API Reference

> Part of the [Orca](../README.md) meta-learning platform.

---

## Overview

The Orca platform exposes four independent HTTP services:

| Service   | Default Port | Base URL              | Description                                        |
|-----------|-------------|-----------------------|----------------------------------------------------|
| OrcaMind  | `8000`       | `http://localhost:8000` | Meta-learning engine and recommendations           |
| OrcaLab   | `8001`       | `http://localhost:8001` | Experiment orchestration and search                |
| OrcaNet   | `8002`       | `http://localhost:8002` | Cross-domain knowledge transfer agent              |
| Orca Web  | `8003`       | `http://localhost:8003` | Backend for Frontend — auth, dashboard, users, service proxies |

All services auto-generate interactive API docs at `GET /docs` (Swagger UI) and `GET /redoc`.

Backend services (OrcaMind, OrcaLab, OrcaNet) use the `/api/v1/` prefix and have no authentication. Orca Web provides JWT-based authentication, proxies backend services for the browser via dedicated proxy routers, and aggregates dashboard data.

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

## OrcaNet API — port 8002

OrcaNet exposes a FastAPI HTTP service for cross-domain transfer scoring, LLM-powered recommendations, task retrieval, and domain-invariant embedding. All endpoints are live and covered by integration tests.

The `X-LLM-Provider` request header can be set to `openai`, `anthropic`, or `local` on any endpoint that invokes the reasoning agent (`/transfer/recommend`, `/explain`) to override the default LLM provider configured at startup without restarting the service. Values outside this set are rejected with **400**.

### Root and health

#### `GET /` — Service info

```json
{ "name": "OrcaNet", "version": "1.0.0", "status": "ok" }
```

---

#### `GET /health` — Dependency health check

Checks OrcaMind (`GET <orcamind_url>/health`, 3 s timeout) and OrcaLab (`GET <orcalab_url>/health`, 3 s timeout) in parallel. The LLM backend check (`agent.llm.ainvoke("ping")`, 5 s timeout) is **opt-in** to avoid recurring token costs from frequent load-balancer probes.

**Query parameters**

| Parameter | Type   | Default | Description                                        |
|-----------|--------|---------|----------------------------------------------------|
| `deep`    | `bool` | `false` | When `true`, also probes the LLM backend           |

**Response**

```json
{
  "status":   "healthy | degraded",
  "orcamind": true,
  "orcalab":  true,
  "llm":      true
}
```

`llm` is `null` when `deep=false` (the LLM check was not performed). `status` is `"degraded"` when any checked component is unreachable; the endpoint always returns **200**.

**Examples**

```http
GET /health          → shallow probe; llm: null
GET /health?deep=true → full probe; llm: true|false
```

---

### Transfer

#### `POST /api/v1/transfer/score` — Score transfer between two tasks

**Request body** — `TransferScoreRequest`

```json
{
  "source_task_id": "uuid-string",
  "target_task_id": "uuid-string",
  "strategy":       "feature"
}
```

| Field            | Type     | Default     | Valid values                              |
|------------------|----------|-------------|-------------------------------------------|
| `source_task_id` | `string` | —           | UUID of a registered task                 |
| `target_task_id` | `string` | —           | UUID of a registered task                 |
| `strategy`       | `string` | `"feature"` | `"feature"`, `"weight"`, `"architecture"` |

**Response** or **400** (unknown strategy) / **404** (task not found)

```json
{
  "overall":            0.72,
  "layer_scores":       { "layer1": 0.80, "layer2": 0.64 },
  "recommended_layers": ["layer1"],
  "reasoning":          "CKA analysis: 1/2 layers exceed threshold 0.5.",
  "strategy":           "feature"
}
```

---

#### `POST /api/v1/transfer/recommend` — LLM agent transfer recommendation

Runs the `OrcaNetAgent` reasoning loop and returns a structured recommendation. Supports the `X-LLM-Provider` header override.

**Request body** — `TransferRecommendRequest`

```json
{
  "target_task_id":    "uuid-string",
  "query_description": "Image classification on small labelled dataset",
  "top_k":             3
}
```

**Response** — `TransferRecommendationResponse` or **502** (LLM agent error)

```json
{
  "top_sources": [
    {
      "task_id":          "uuid-string",
      "task_name":        "brain MRI classification",
      "similarity_score": 0.88,
      "transfer_score":   0.76,
      "reasoning":        "High feature overlap in early layers."
    }
  ],
  "recommended_strategy": "feature",
  "expected_improvement": 0.14,
  "explanation":          "Feature transfer is recommended due to shared low-level visual structure.",
  "confidence":           0.82
}
```

---

#### `GET /api/v1/transfer/{mapping_id}` — Get stored transfer mapping

**Response** — `TransferMapping` or **404** / **422** (invalid UUID)

```json
{
  "mapping_id":     "uuid",
  "source_task_id": "uuid",
  "target_task_id": "uuid",
  "transfer_score": 0.65,
  "transfer_type":  "feature",
  "metadata":       null,
  "created_at":     "datetime"
}
```

---

#### `POST /api/v1/transfer/validate` — Three-way pipeline validation

Runs the full `TransferPipeline`: scores the transfer, optionally triggers an OrcaLab validation experiment, and persists a `TransferMapping` to the database.

**Request body** — `TransferValidateRequest`

```json
{
  "source_task_id": "uuid-string",
  "target_task_id": "uuid-string",
  "strategy":       "feature",
  "validate":       true
}
```

| Field            | Type      | Default    | Description                                                              |
|------------------|-----------|------------|--------------------------------------------------------------------------|
| `source_task_id` | `string`  | —          | UUID of the source task (model donor)                                    |
| `target_task_id` | `string`  | —          | UUID of the target task (model recipient)                                |
| `strategy`       | `string`  | `"feature"` | Transfer strategy — `"feature"`, `"weight"`, or `"architecture"`        |
| `validate`       | `boolean` | `true`     | When `true` and `score.overall > 0.4`, triggers an OrcaLab experiment   |

**Response** — `200 OK` — `TransferValidateResponse`

```json
{
  "score": {
    "overall":             0.78,
    "layer_scores":        { "layer0": 0.78 },
    "recommended_layers":  ["layer0"],
    "reasoning":           "High CKA alignment across shared layers"
  },
  "experiment_result": {
    "experiment_id": "uuid",
    "task_id":       "uuid",
    "model_id":      null,
    "status":        "completed",
    "metrics":       { "accuracy": 0.90, "baseline_accuracy": 0.78 }
  },
  "mapping": {
    "mapping_id":     "uuid",
    "source_task_id": "uuid",
    "target_task_id": "uuid",
    "transfer_score": 0.78,
    "transfer_type":  "feature",
    "metadata":       null,
    "created_at":     "datetime"
  },
  "improvement_over_baseline": 0.12
}
```

`experiment_result` is `null` when `validate=false`, when the score is at or below the 0.4 threshold, or when the OrcaLab experiment times out (3600 s). `improvement_over_baseline` is `null` whenever `experiment_result` is `null` or when `accuracy`/`baseline_accuracy` are absent from the metrics.

**Error responses**

| Code | Condition |
|------|-----------|
| 400  | Unknown `strategy` value |
| 404  | `source_task_id` or `target_task_id` not found |
| 503  | OrcaMind is unreachable or returned a server error (`ServiceUnavailableError`) |

---

### Retrieval

#### `POST /api/v1/retrieve` — Retrieve similar tasks

Runs three-stage hybrid retrieval (FAISS → metadata filter → optional LLM re-rank). When `query_description` is provided, queries are expanded via `QueryExpander` before FAISS search.

**Request body** — `RetrieveRequest`

```json
{
  "task_id":           "uuid-string",
  "query_description": "optional natural-language description",
  "filters":           { "domain": "vision" },
  "top_k":             10
}
```

| Field               | Type             | Default | Description                                          |
|---------------------|------------------|---------|------------------------------------------------------|
| `task_id`           | `string`         | —       | UUID of the query task                               |
| `query_description` | `string \| null` | `null`  | Enables LLM query expansion when provided            |
| `filters`           | `object \| null` | `null`  | Field-equality filters applied to retrieved tasks    |
| `top_k`             | `int`            | `10`    | Maximum number of results to return                  |

**Response** — `list[SimilarityResult]` or **404** (task not found)

```json
[
  { "task_id": "uuid", "score": 0.91, "rank": 1 },
  { "task_id": "uuid", "score": 0.84, "rank": 2 }
]
```

---

### Explain

#### `POST /api/v1/explain` — Generate transfer explanation

Invokes the `OrcaNetAgent` with an explanation-focused query and returns the natural-language explanation from the agent's `TransferRecommendationResponse`. Supports the `X-LLM-Provider` header override.

**Request body** — `ExplainRequest`

```json
{
  "source_task_id": "uuid-string",
  "target_task_id": "uuid-string",
  "strategy":       "feature"
}
```

**Response** or **502** (LLM agent error / unparseable response)

```json
{
  "explanation": "Feature transfer is recommended due to shared low-level visual structure between the source retinal scan task and the target fundus classification task."
}
```

---

### Embed

#### `POST /api/v1/cross-domain-embed` — Compute domain-invariant embedding

Produces a 64-dim L2-normalised embedding from the `CrossDomainEmbedder` (DANN-based). Accepts either a registered task ID (task features are derived from its registry metadata) or a raw 25-dim feature vector.

**Request body** — `EmbedRequest`

```json
{
  "task_id":             "uuid-string",
  "statistical_features": null,
  "description":          null
}
```

Exactly one of `task_id` or `statistical_features` must be provided; omitting both **or** supplying both returns **422**.

| Field                  | Type                    | Description                                                         |
|------------------------|-------------------------|---------------------------------------------------------------------|
| `task_id`              | `string \| null`        | UUID of a registered task; features derived from DB                 |
| `statistical_features` | `list[float] \| null`   | Pre-computed feature vector — **must be exactly 25 elements**       |
| `description`          | `string \| null`        | Reserved; not used in current embedding computation                 |

**Response** — `EmbedResponse` or **404** (task not found) / **422** (validation error)

```json
{ "embedding": [0.021, -0.133, ..., 0.047] }
```

The returned list has exactly 64 elements and is L2-normalised.

---

## Orca Web BFF API — port 8003

Orca Web is the Backend for Frontend (BFF) that provides JWT authentication, dashboard aggregation, user management, and service proxy routers. All endpoints are served under `root_path="/api/v1"`. The BFF proxies OrcaMind, OrcaLab, and OrcaNet for the browser via dedicated proxy routers that inject an `X-Orca-User-ID` header and log mutating operations to the activity log.

### Health

#### `GET /health` — BFF health check (no auth)

Checks Postgres, Redis, OrcaMind, OrcaLab, and OrcaNet in parallel.

**Response** — `200 OK` (all healthy) or `503 Service Unavailable` (any degraded)

```json
{
  "status": "healthy",
  "services": {
    "postgres": true,
    "redis": true,
    "orcamind": true,
    "orcalab": true,
    "orcanet": true
  }
}
```

---

### Authentication

#### `POST /auth/register` — Register a new user

Status: **201 Created**

**Request body**

```json
{
  "email": "alice@example.com",
  "username": "alice",
  "password": "secret"
}
```

**Response** — `TokenResponse`

```json
{ "access_token": "eyJ...", "token_type": "bearer" }
```

Sets an `httponly` refresh token cookie scoped to `/api/v1/auth`.

---

#### `POST /auth/login` — Authenticate with email and password

**Response** — `TokenResponse` or **401**

---

#### `POST /auth/refresh` — Rotate refresh token

Reads the `refresh_token` cookie, revokes the old session, and issues a new access + refresh token pair.

**Response** — `TokenResponse` or **401**

---

#### `POST /auth/logout` — Revoke refresh token

Status: **204 No Content**

Revokes the current refresh session and clears the cookie.

---

#### `GET /auth/me` — Get current user profile (auth required)

**Response** — `UserResponse`

```json
{
  "user_id": "uuid",
  "email": "alice@example.com",
  "username": "alice",
  "role": "user",
  "preferences": null
}
```

---

#### `PATCH /auth/me` — Update current user profile (auth required)

**Request body** — `ProfileUpdate`

```json
{ "username": "alice_updated", "preferences": {"theme": "dark"} }
```

**Response** — `UserResponse`

---

#### `GET /auth/oauth/{provider}` — Redirect to OAuth provider

Supports `google` and `github`. Returns a redirect response.

---

#### `GET /auth/oauth/{provider}/callback` — Handle OAuth callback

**Response** — `TokenResponse`

---

### Dashboard

#### `GET /dashboard/overview` — Aggregated stats (auth required)

Proxies OrcaMind, OrcaLab, and OrcaNet to return combined service stats.

---

#### `GET /dashboard/stats` — Public stats (no auth)

Returns public counts for the landing page.

---

### Users

#### `GET /users/{user_id}` — Get user profile (auth required)

Callers may only access their own profile unless they have the `admin` role.

**Response** — `UserResponse` or **403** / **404**

---

### OrcaMind Proxy (auth required)

All OrcaMind proxy endpoints require JWT authentication. The BFF forwards the request to the upstream OrcaMind service, injecting an `X-Orca-User-ID` header. Query parameters are forwarded with full multi-value support (e.g. `?tag=a&tag=b` preserves both values). Connection errors return 502; timeouts (10 s) return 504. POST endpoints log activity to the `activity_log` table.

#### `GET /orcamind/tasks` — List tasks

Proxies to `GET {ORCAMIND}/api/v1/tasks`. Query parameters (`limit`, `domain`, etc.) are forwarded.

---

#### `GET /orcamind/tasks/{task_id}` — Get task

Proxies to `GET {ORCAMIND}/api/v1/tasks/{task_id}`.

---

#### `POST /orcamind/tasks` — Embed a new task

Proxies to `POST {ORCAMIND}/api/v1/tasks/embed`. Logs `task_created` activity.

---

#### `POST /orcamind/recommend` — Request model recommendations

Proxies to `POST {ORCAMIND}/api/v1/recommend-model`. Logs `model_recommended` activity.

---

#### `POST /orcamind/similar-tasks` — Find similar tasks

Proxies to `POST {ORCAMIND}/api/v1/similar-tasks`. Logs `similar_tasks_searched` activity.

---

#### `POST /orcamind/predict-performance` — Predict model performance

Proxies to `POST {ORCAMIND}/api/v1/predict-performance`. Logs `performance_predicted` activity.

---

### OrcaLab Proxy (auth required)

All OrcaLab proxy endpoints require JWT authentication. Same error handling and header injection as OrcaMind proxy.

#### `GET /orcalab/experiments` — List experiments

Proxies to `GET {ORCALAB}/api/v1/experiments`. Query parameters forwarded.

---

#### `GET /orcalab/experiments/{experiment_id}` — Get experiment

Proxies to `GET {ORCALAB}/api/v1/experiments/{experiment_id}`.

---

#### `POST /orcalab/experiments` — Create experiment

Proxies to `POST {ORCALAB}/api/v1/experiments`. Logs `experiment_started` activity.

---

#### `POST /orcalab/sweeps` — Start hyperparameter sweep

Proxies to `POST {ORCALAB}/api/v1/sweeps`. Logs `sweep_started` activity.

---

#### `GET /orcalab/sweeps/{sweep_id}` — Get sweep status

Proxies to `GET {ORCALAB}/api/v1/sweeps/{sweep_id}`.

---

#### `WS /orcalab/ws/experiments/{experiment_id}/live` — Live experiment metrics stream

Authenticated WebSocket proxy that relays real-time experiment metrics between the browser and OrcaLab. Since browsers cannot set `Authorization` headers on WebSocket connections, the JWT access token is passed as a `token` query parameter.

**Authentication:** Pass `?token=<jwt_access_token>` as a query parameter. The server validates the token and extracts the user ID before accepting the connection. Invalid or missing tokens result in close code **4001**.

**Connection flow:**

1. Browser opens WebSocket to the BFF with `?token=<jwt>`
2. BFF validates the JWT and extracts the user ID
3. BFF opens an upstream WebSocket to OrcaLab at `ws://{ORCALAB}/api/v1/experiments/{experiment_id}/live`
4. Messages are relayed bidirectionally until either side disconnects

**Message directions:**

| Direction | Content | Format |
|-----------|---------|--------|
| OrcaLab → Browser | Metric updates (epoch, loss, status) | JSON text frames |
| Browser → OrcaLab | Control messages (pause, resume, cancel) | JSON text frames |

**Metric update frame** (OrcaLab → Browser):

```json
{
  "experiment_id": "uuid-string",
  "status": "running",
  "epoch": 3,
  "loss": 0.4721,
  "metrics": { "loss": 0.4721, "epoch": 3 }
}
```

**Control message frame** (Browser → OrcaLab):

```json
{ "action": "pause" }
```

**Error handling:**

| Condition | Behaviour |
|-----------|-----------|
| Missing or invalid JWT | Close with code **4001** (connection never accepted) |
| Upstream connection failure (refused, DNS, timeout) | Send `{"error": "upstream_unavailable"}` and close |
| Upstream disconnect (experiment completed/failed) | Close browser connection normally |
| Browser disconnect | Close upstream connection |

**Implementation notes:**

- A 30-second heartbeat ping detects stale upstream connections
- Both relay directions run as concurrent asyncio tasks
- Uses the `websockets` library for the upstream connection
- Binary upstream messages are decoded to text before forwarding

---

### OrcaNet Proxy (auth required)

All OrcaNet proxy endpoints require JWT authentication. Same error handling and header injection as OrcaMind proxy.

#### `POST /orcanet/transfer/score` — Score transfer

Proxies to `POST {ORCANET}/api/v1/transfer/score`. Logs `transfer_scored` activity.

---

#### `POST /orcanet/transfer/recommend` — Get transfer recommendations

Proxies to `POST {ORCANET}/api/v1/transfer/recommend`. Logs `transfer_recommended` activity.

---

#### `POST /orcanet/retrieve` — Retrieve similar tasks

Proxies to `POST {ORCANET}/api/v1/retrieve`. Logs `tasks_retrieved` activity.

---

#### `POST /orcanet/explain` — Explain transfer

Proxies to `POST {ORCANET}/api/v1/explain`. Logs `transfer_explained` activity.

---

### Proxy Error Responses

All proxy endpoints return the same error format on upstream failures:

| Status | Body | Condition |
|--------|------|-----------|
| `502` | `{"detail": "Service unavailable"}` | Upstream connection refused or DNS failure |
| `504` | `{"detail": "Service timeout"}` | Upstream did not respond within 10 seconds |

On success, the response mirrors the upstream's status code, body, and content-type exactly.

---

## Health Endpoints

All four services expose a health endpoint with no authentication requirement.

| Service  | Endpoint      | `status` field          | Additional fields                                     |
|----------|---------------|-------------------------|-------------------------------------------------------|
| OrcaMind | `GET /health` | `"healthy" \| "degraded"` | `"db": bool`, `"faiss": bool`, `"mlflow": bool`     |
| OrcaLab  | `GET /health` | `"healthy" \| "degraded"` | `"db": bool`, `"prefect": bool`                     |
| OrcaNet  | `GET /health` | `"healthy" \| "degraded"` | `"orcamind": bool`, `"orcalab": bool`, `"llm": bool \| null` (null unless `?deep=true`) |

The OrcaMind health endpoint reports `faiss: false` when the FAISS index has not been built yet. The service remains fully operational — only `/recommend-model` and `/similar-tasks` return 503 until the index is populated.

The OrcaNet health endpoint is always **200**. When one or more checked components are unreachable, `status` is `"degraded"` but the individual flag for that component is `false`, allowing callers to determine exactly which dependency is unavailable. The `llm` field is `null` for shallow probes and `true|false` only when `?deep=true` is passed.

---

## OrcaNet Transfer Python API

The transfer subpackage provides a pure-Python API for three concrete transfer strategies: CKA-based feature-level scoring (`FeatureTransfer`), direct parameter-tensor transfer with selective layer matching (`WeightTransfer`), and architecture graph-embedding similarity with config adaptation (`ArchitectureTransfer`). None of the strategies expose an HTTP endpoint in the current release — they are consumed directly by OrcaNet's internal recommendation pipeline.

### `linear_cka(X, Y) -> float`

```python
from orcanet.transfer import linear_cka
```

Compute linear Centered Kernel Alignment (Kornblith et al. 2019) between two activation matrices.

| Parameter | Type | Description |
|---|---|---|
| `X` | `np.ndarray (n, p)` | Activation matrix for one model; `n` samples, `p` features |
| `Y` | `np.ndarray (n, q)` | Activation matrix for the other model; same `n` samples, `q` features |

**Returns** `float` in [0, 1]. A value of 1.0 indicates identical representational geometry; 0.0 indicates orthogonal representations.

### `TransferStrategy` (ABC)

```python
from orcanet.transfer import TransferStrategy
```

Abstract base class for all transfer strategies. Subclasses must implement:

| Method | Signature | Description |
|---|---|---|
| `score_transfer` | `(source: Task, target: Task) -> TransferScore` | Compute transfer score between two registered models |
| `execute_transfer` | `(source: Task, target: Task, source_model: nn.Module) -> nn.Module` | Apply transfer to produce an adapted model |
| `get_transfer_metadata` | `() -> dict` | Return strategy configuration for logging |

### `TransferScore`

```python
from orcanet.transfer import TransferScore
```

Dataclass returned by `score_transfer`. **Note:** this is the internal rich type; `orca_shared.schemas.transfer.TransferScore` is the lightweight API schema for inter-service exchange.

| Field | Type | Description |
|---|---|---|
| `overall` | `float` | Aggregate transferability score in [0, 1]. `FeatureTransfer`: depth-weighted mean CKA. `WeightTransfer`: `n_matched / n_total` parameter ratio. `ArchitectureTransfer`: max cosine similarity across registered candidate configs. `MultiTaskTransfer`: cosine similarity of `CrossDomainEmbedder` embeddings, or `0.5` when no task features are registered. |
| `layer_scores` | `dict[str, float]` | Per-named-layer score. `FeatureTransfer`: CKA similarity value. `WeightTransfer`: `1.0` (matched) or `0.0` (unmatched) per parameter tensor. `ArchitectureTransfer`: cosine similarity per registered candidate config name. `MultiTaskTransfer`: `{"cosine_similarity": float}` when features registered; `{}` otherwise. |
| `recommended_layers` | `list[str]` | Layers selected for transfer. `FeatureTransfer`: layers whose CKA exceeds `cka_threshold`. `WeightTransfer`: all matched parameter names. `ArchitectureTransfer`: always `[]` (architecture-level decision). `MultiTaskTransfer`: `["backbone"]` when similarity > 0.5; `[]` otherwise. |
| `reasoning` | `str` | Human-readable summary, e.g. `"CKA analysis: 3/4 layers exceed threshold 0.5."`, `"Matched 4/4 layers by name"`, `"Architecture mlp_128_64 is most similar to source architecture"`, or `"Multi-task training beneficial: similarity 0.73 > threshold 0.5"`. |

### `FeatureTransfer`

```python
from orcanet.transfer import FeatureTransfer
```

Concrete `TransferStrategy` using per-layer linear CKA to score transferability and selectively patch weights.

#### Constructor

```python
FeatureTransfer(
    orcamind_client: OrcaMindClient | None = None,
    probe_data: np.ndarray | None = None,
    cka_threshold: float = 0.5,
)
```

| Parameter | Default | Description |
|---|---|---|
| `orcamind_client` | `None` | Optional client stored for future registry-backed model resolution; not used in the current scoring path |
| `probe_data` | `None` | Shared input array `(n_samples, input_dim)` passed through both models to collect activations. Required before calling `score_transfer`. |
| `cka_threshold` | `0.5` | Minimum per-layer CKA for a layer to appear in `recommended_layers` |

#### Methods

**`register_model(task_id: str, model: nn.Module) -> None`**

Register an `nn.Module` under `task_id`. Must be called for both source and target tasks before `score_transfer`.

**`score_transfer(source: Task, target: Task) -> TransferScore`**

Collect activations from both registered models on `probe_data`, compute per-layer CKA, and return a `TransferScore`. Raises `ValueError` if either model is not registered or if `probe_data` is `None`.

**`execute_transfer(source: Task, target: Task, source_model: nn.Module) -> nn.Module`**

Clone the registered target model and patch in weights from `source_model` for all `recommended_layers`. Non-recommended layers retain the target model's initialisation. Raises `ValueError` if the target task has no registered model.

**`get_transfer_metadata() -> dict`**

Return `{"strategy": "feature_cka", "cka_threshold": float, "n_registered_models": int, "has_probe_data": bool}`.

#### Example

```python
import numpy as np
import torch.nn as nn
from orcanet.transfer import FeatureTransfer

probe = np.random.default_rng(0).standard_normal((100, 10)).astype("float32")
transfer = FeatureTransfer(probe_data=probe, cka_threshold=0.6)

transfer.register_model(str(source_task.task_id), source_model)
transfer.register_model(str(target_task.task_id), target_model)

score = transfer.score_transfer(source_task, target_task)
print(score.overall)           # e.g. 0.87
print(score.recommended_layers)  # e.g. ['0', '0.0', '0.2']

if score.overall > 0.4:
    adapted = transfer.execute_transfer(source_task, target_task, source_model)
    # adapted is ready for fine-tuning on the target task
```

### `WeightTransfer`

```python
from orcanet.transfer import WeightTransfer
```

Concrete `TransferStrategy` that transfers model weights by matching parameter tensors between a source and target model. Matching can be performed by parameter name, tensor shape, or both. Unmatched tensors are reinitialised in the returned model. Pairs with `get_optimizer_with_layer_lr` to apply decayed learning rates to transferred layers during fine-tuning.

#### Constructor

```python
WeightTransfer(
    match_by: str = "name",        # "name" | "shape" | "both"
    frozen_epochs: int = 5,
    layer_lr_decay: float = 0.1,
)
```

| Parameter | Default | Description |
|---|---|---|
| `match_by` | `"name"` | Matching criterion. `"name"`: match if the parameter name exists in the source state dict (regardless of shape). `"shape"`: match if any source parameter has the same tensor shape. `"both"`: match only when name exists **and** shapes agree — the strictest mode, safe for cross-architecture transfer. |
| `frozen_epochs` | `5` | Number of epochs to freeze transferred layers after transfer. Stored for downstream use; not enforced internally by this class. |
| `layer_lr_decay` | `0.1` | Default decay factor for `get_optimizer_with_layer_lr`. Transferred layers receive `base_lr * layer_lr_decay`; new layers receive `base_lr`. |

Raises `ValueError` on construction if `match_by` is not one of `"name"`, `"shape"`, `"both"`.

#### Methods

**`register_model(task_id: str, model: nn.Module) -> None`**

Register an `nn.Module` under `task_id`. Must be called for both source and target tasks before `score_transfer`.

**`score_transfer(source: Task, target: Task) -> TransferScore`**

Iterate the target model's `state_dict()`, call `_matches` for each parameter tensor, and return a `TransferScore` with:

- `overall` = `n_matched / n_total`
- `layer_scores` = `{param_name: 1.0 | 0.0}`
- `recommended_layers` = all matched parameter names
- `reasoning` = `f"Matched {n_matched}/{n_total} layers by {match_by}"`

Raises `ValueError` if either model is not registered.

**`execute_transfer(source: Task, target: Task, source_model: nn.Module) -> nn.Module`**

Deep-copy the registered target model to create the adapted base, then for each parameter tensor:

- If a safe source tensor can be resolved (name match + shape agreement, or shape-first scan for `match_by="shape"`): copy it in-place via `tensor.copy_()`.
- Otherwise: reinitialise with `kaiming_uniform_` (2-D+ tensors, e.g. weight matrices) or `zeros_` (1-D tensors, e.g. bias vectors).

Returns the adapted `nn.Module`. The names of transferred parameters are stored in `transfer.last_transferred` after the call; pass this list directly to `get_optimizer_with_layer_lr`. Raises `ValueError` if the target task has no registered model.

Shape mismatches are always handled silently — `copy_` is never called with incompatible shapes.

**`get_transfer_metadata() -> dict`**

Return `{"strategy": "weight_transfer", "match_by": str, "frozen_epochs": int, "layer_lr_decay": float}`.

#### Example

```python
import torch.nn as nn
from orcanet.transfer import WeightTransfer, get_optimizer_with_layer_lr

transfer = WeightTransfer(match_by="both", layer_lr_decay=0.1)

transfer.register_model(str(source_task.task_id), source_model)
transfer.register_model(str(target_task.task_id), target_model)

score = transfer.score_transfer(source_task, target_task)
print(score.overall)            # e.g. 0.5 (2/4 params matched for different out_dim)
print(score.recommended_layers) # e.g. ['0.weight', '0.bias']

adapted = transfer.execute_transfer(source_task, target_task, source_model)
# adapted: nn.Module — registered target architecture with source weights patched in
# transfer.last_transferred: list[str] — e.g. ['0.weight', '0.bias', '2.weight', '2.bias']

optimizer = get_optimizer_with_layer_lr(
    adapted,
    transferred_layers=transfer.last_transferred,
    base_lr=1e-3,
    decay=0.1,  # transferred layers get lr=1e-4; new layers get lr=1e-3
)
```

### `get_optimizer_with_layer_lr`

```python
from orcanet.transfer import get_optimizer_with_layer_lr
```

Module-level utility that builds a `torch.optim.Adam` optimizer with per-parameter learning-rate groups. Transferred layers receive a decayed rate to allow fine adjustment of already-learned features; new or reinitialised layers receive the full base rate to learn from scratch.

```python
get_optimizer_with_layer_lr(
    model: nn.Module,
    transferred_layers: list[str],
    base_lr: float,
    decay: float = 0.1,
) -> torch.optim.Adam
```

| Parameter | Description |
|---|---|
| `model` | The adapted model returned by `WeightTransfer.execute_transfer`. |
| `transferred_layers` | Parameter names (as returned by `model.named_parameters()`) that were copied from the source. Use `WeightTransfer.last_transferred` set by the preceding `execute_transfer` call. |
| `base_lr` | Learning rate for non-transferred (new / reinitialised) parameters. |
| `decay` | Multiplicative factor applied to `base_lr` for transferred parameters. Default `0.1`. |

**Returns** `torch.optim.Adam` with one parameter group per `named_parameters()` entry.

### `ArchitectureTransfer`

```python
from orcanet.transfer import ArchitectureTransfer
```

Concrete `TransferStrategy` that recommends and adapts model architectures for a target domain by comparing architecture graph embeddings. Uses the OrcaMind service to retrieve the source task's best-known architecture name, then scores all locally registered candidate configs by cosine similarity via `ArchitectureEmbedder` and returns the best match. Adapts the winning config to the target task's input/output dimensions and builds an `nn.Sequential` with middle-layer weights optionally copied from the source model.

#### Constructor

```python
ArchitectureTransfer(
    architecture_embedder: ArchitectureEmbedder,
    orcamind_client: OrcaMindClient,
    top_k_candidates: int = 10,
)
```

| Parameter | Default | Description |
|---|---|---|
| `architecture_embedder` | — | `ArchitectureEmbedder` instance used to compute cosine similarity between architecture configs |
| `orcamind_client` | — | Async `OrcaMindClient` that resolves the source task's best model name |
| `top_k_candidates` | `10` | Stored for metadata reporting; all registered configs are always scored |

#### Methods

**`register_config(name: str, config: ArchConfig) -> None`**

Register a named architecture config for similarity scoring. `ArchConfig` is `dict[str, Any]` with `"input_dim": int` and `"layers": list` keys. Re-registering an existing name overwrites it.

**`score_transfer(source: Task, target: Task) -> TransferScore`**

Fetch the source task's best architecture from OrcaMind, compute cosine similarity for every registered candidate, and return a `TransferScore` with:

- `overall` = highest candidate similarity (clamped to 0)
- `layer_scores` = `{candidate_name: similarity}` for every registered config
- `recommended_layers` = `[]` (architecture-level decision — no per-layer concept)
- `reasoning` = human-readable string naming the best candidate

Returns `overall == 0.0` when no configs are registered.

**`execute_transfer(source: Task, target: Task, source_model: nn.Module) -> nn.Module`**

1. Calls `score_transfer` if it has not been called yet.
2. Deep-copies the best candidate config and updates `input_dim` / last-layer `size` for the target task via `adapt_architecture`.
3. Builds an `nn.Sequential` with `kaiming_uniform_` weight initialisation.
4. Copies middle-layer weights from `source_model` where shapes match (first and last linear layers are never copied).

Returns the adapted `nn.Module`. Does not mutate `source_model`.

**`get_transfer_metadata() -> dict`**

Returns `{"strategy": "architecture_transfer", "top_k_candidates": int, "n_registered_configs": int}`.

### `adapt_architecture`

```python
from orcanet.transfer import adapt_architecture
```

Module-level pure function that returns a deep-copied architecture config with its boundary dimensions updated for a target task.

```python
adapt_architecture(
    config: ArchConfig,
    target_task: Task,
) -> ArchConfig
```

| Parameter | Description |
|---|---|
| `config` | Source architecture config with `"input_dim"` and `"layers"` keys |
| `target_task` | Task whose `n_features` (new input dim) and `n_classes` (new output size) are applied |

Only `config["input_dim"]` and the last layer's `"size"` are modified; all hidden layers are preserved. If `target_task.n_features` or `target_task.n_classes` is `None`, the corresponding dimension is left unchanged.

#### Example

```python
from orcanet.transfer import ArchitectureTransfer, adapt_architecture

transfer = ArchitectureTransfer(
    architecture_embedder=embedder,
    orcamind_client=client,
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
print(score.overall)            # e.g. 0.87
print(score.recommended_layers) # []  (architecture-level — no per-layer selection)

if score.overall > 0.4:
    adapted = transfer.execute_transfer(source_task, target_task, source_model)
    # adapted: nn.Sequential with in_features == target_task.n_features
    #           and out_features == target_task.n_classes

# adapt_architecture can also be called standalone
new_config = adapt_architecture(
    {"input_dim": 25, "layers": [{"type": "linear", "size": 10, "activation": "none"}]},
    target_task,   # n_features=50, n_classes=3
)
# new_config["input_dim"] == 50
# new_config["layers"][-1]["size"] == 3
```

### `MultiTaskModel`

```python
from orcanet.transfer import MultiTaskModel
```

`nn.Module` subclass returned by `MultiTaskTransfer.execute_transfer`. Routes inputs to a task-specific head after passing them through a shared backbone.

#### Constructor

```python
MultiTaskModel(
    backbone: nn.Module,
    task_heads: dict[str, nn.Module],
    task_weighting: str = "equal",
    log_sigmas: dict[str, nn.Parameter] | None = None,
)
```

| Parameter | Default | Description |
|---|---|---|
| `backbone` | — | Shared feature extractor, registered as a submodule |
| `task_heads` | — | Per-task output heads; stored as `nn.ModuleDict` to ensure heads appear in `model.parameters()` and `model.state_dict()` |
| `task_weighting` | `"equal"` | Weighting scheme string, stored for reference; does not affect `forward` |
| `log_sigmas` | `None` | Per-task learnable log-variance `nn.Parameter` objects for uncertainty weighting; stored as `nn.ParameterDict`. Pass `None` or an empty dict when not using uncertainty weighting. |

#### Methods

**`forward(x: Tensor, task_id: str) -> Tensor`**

Passes `x` through `backbone`, then routes the result through `task_heads[task_id]`. Raises `KeyError` for unregistered task ids.

**`compute_loss(batch: dict[str, tuple[Tensor, Tensor]], weights: dict[str, float]) -> Tensor`**

Computes a weighted sum of per-task cross-entropy losses: `Σ weights[tid] · CE(forward(x, tid), y)`. For use with `"equal"` or `"gradnorm"` weighting. Returns a scalar tensor.

**`compute_uncertainty_loss(batch: dict[str, tuple[Tensor, Tensor]]) -> Tensor`**

Implements the Kendall et al. 2018 multi-task loss: `L = Σ exp(−2·log_σᵢ) · CEᵢ + log_σᵢ`. Requires `log_sigmas` to be populated (set by `MultiTaskTransfer` when `task_weighting="uncertainty"`). Gradients flow into `log_sigmas` automatically. Returns a scalar tensor.

### `MultiTaskTransfer`

```python
from orcanet.transfer import MultiTaskTransfer
```

Concrete `TransferStrategy` for joint training across multiple tasks with a shared backbone.

#### Constructor

```python
MultiTaskTransfer(
    backbone: nn.Module,
    task_weighting: str = "equal",        # "equal" | "uncertainty" | "gradnorm"
    task_head_hidden_dim: int = 64,
    embedder: CrossDomainEmbedder | None = None,
)
```

| Parameter | Default | Description |
|---|---|---|
| `backbone` | — | Shared feature extractor. Output dimensionality is inferred automatically from the last `nn.Linear` in the backbone. |
| `task_weighting` | `"equal"` | Weighting scheme. `"equal"`: uniform `1/n`; `"uncertainty"`: Kendall et al. 2018 learnable log-variance; `"gradnorm"`: caller-driven gradient-norm renormalisation. |
| `task_head_hidden_dim` | `64` | Hidden size of each two-layer task head. |
| `embedder` | `None` | `CrossDomainEmbedder` used to embed task feature tensors in `score_transfer`. A default instance (`input_dim=25`) is created when not supplied. |

Raises `ValueError` on construction if `task_weighting` is not one of `"equal"`, `"uncertainty"`, `"gradnorm"`.

#### Methods

**`add_task(task: Task, head_output_dim: int) -> None`**

Create a `nn.Sequential(Linear(backbone_out, hidden), ReLU(), Linear(hidden, head_output_dim))` head and register it under `str(task.task_id)`. Automatically rebalances `_task_weights` after registration. For `"uncertainty"` weighting, also creates a `nn.Parameter(torch.zeros(1))` log-sigma for the task.

Raises `ValueError` if `task.task_id` has already been registered, preventing silent overwrite of trained parameters.

**`register_task_features(task_id: str, features: Tensor) -> None`**

Store a meta-feature tensor for the given task id (shape `(1, embedder_input_dim)` or `(embedder_input_dim,)`). Must be called before `score_transfer` to obtain a meaningful similarity score. The tensor is detached from any autograd graph on storage so that calling this inside a training loop does not retain intermediate activations.

**`score_transfer(source: Task, target: Task) -> TransferScore`**

Compute embedding cosine similarity if features are registered for both tasks; otherwise return a neutral `TransferScore(overall=0.5)`. The reasoning string uses the wording `"Multi-task training beneficial: similarity {:.2f} > threshold 0.5"` when `overall > 0.5`.

**`execute_transfer(source: Task, target: Task, source_model: nn.Module) -> nn.Module`**

Auto-registers any unregistered task using `task.n_classes` (or `1` if absent) as the head output dimension, then returns a `MultiTaskModel` bundling the shared backbone with all registered heads and log-sigma parameters. The returned model is ready for immediate training.

**`update_gradnorm_weights(grad_norms: dict[str, float]) -> None`**

Renormalise `_task_weights` for the task ids present in `grad_norms`. Only those tasks are updated — omitted tasks retain their existing weight exactly. The updated tasks are scaled so that their combined weight equals what they held before the call, keeping the global weight sum stable.

Raises `ValueError` if any key in `grad_norms` does not correspond to a registered task id. A no-op when passed an empty dict.

**`task_weights` (property) `-> dict[str, float]`**

Returns a copy of the current per-task weight dictionary. Pass directly to `MultiTaskModel.compute_loss(batch, strategy.task_weights)`.

**`get_transfer_metadata() -> dict`**

Returns `{"strategy": "multi_task_transfer", "task_weighting": str, "task_head_hidden_dim": int, "n_registered_tasks": int, "backbone_out_dim": int}`.

#### Example

```python
import torch
import torch.nn as nn
from orcanet.transfer import MultiTaskTransfer, MultiTaskModel

# Shared backbone
backbone = nn.Sequential(nn.Linear(25, 64), nn.ReLU())

# --- Equal weighting ---
strategy = MultiTaskTransfer(backbone, task_weighting="equal")
strategy.add_task(source_task, head_output_dim=3)
strategy.add_task(target_task, head_output_dim=5)

model: MultiTaskModel = strategy.execute_transfer(source_task, target_task, backbone)
# model.task_heads["<source_uuid>"]  → Linear(64,64) → ReLU → Linear(64,3)
# model.task_heads["<target_uuid>"]  → Linear(64,64) → ReLU → Linear(64,5)

x = torch.randn(16, 25)
batch = {
    str(source_task.task_id): (x, torch.randint(0, 3, (16,))),
    str(target_task.task_id): (x, torch.randint(0, 5, (16,))),
}
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
opt.zero_grad()
model.compute_loss(batch, strategy.task_weights).backward()
opt.step()

# --- Uncertainty weighting (Kendall et al. 2018) ---
strategy_unc = MultiTaskTransfer(backbone, task_weighting="uncertainty")
strategy_unc.add_task(source_task, 3)
strategy_unc.add_task(target_task, 5)

model_unc: MultiTaskModel = strategy_unc.execute_transfer(source_task, target_task, backbone)
# model_unc.log_sigmas["<source_uuid>"]  → nn.Parameter(tensor([0.]))
# model_unc.log_sigmas["<target_uuid>"]  → nn.Parameter(tensor([0.]))

opt_unc = torch.optim.Adam(model_unc.parameters(), lr=1e-3)
opt_unc.zero_grad()
model_unc.compute_uncertainty_loss(batch).backward()
# log_sigmas are updated automatically — noisy tasks learn larger sigma
opt_unc.step()

# --- GradNorm weighting ---
strategy_gn = MultiTaskTransfer(backbone, task_weighting="gradnorm")
strategy_gn.add_task(source_task, 3)
strategy_gn.add_task(target_task, 5)
model_gn = strategy_gn.execute_transfer(source_task, target_task, backbone)

# Compute per-task gradient norms (caller's responsibility), then update
grad_norms = {str(source_task.task_id): 1.2, str(target_task.task_id): 0.6}
strategy_gn.update_gradnorm_weights(grad_norms)
model_gn.compute_loss(batch, strategy_gn.task_weights).backward()
```

---

## OrcaNet Retrieval Python API

The retrieval subpackage provides the three-stage hybrid pipeline as a pure-Python API. All three classes are exported from `orcanet.retrieval`.

```python
from orcanet.retrieval import QueryExpander, LLMRanker, HybridRetriever
```

### `QueryExpander`

Generates alternative phrasings of a task description to broaden FAISS recall.

```python
from orcanet.retrieval import QueryExpander

expander = QueryExpander(llm=my_llm)
alternatives = await expander.expand("brain MRI binary classification", n_expansions=3)
# → ["medical image classification", "neurological imaging task", "3D scan binary classification"]
```

**Constructor**

| Parameter | Type | Description |
|---|---|---|
| `llm` | `langchain_core.language_models.BaseLLM` | Any LangChain-compatible LLM; called via `ainvoke` |

**`async expand(query, n_expansions=3) -> list[str]`**

Builds a prompt asking the LLM for `n_expansions` alternatives, calls `ainvoke`, and strips numbered prefixes / bullet markers from each response line. Handles both `BaseLLM` (returns `str`) and chat-model (returns a message object with `.content`) response types. Returns an empty list when the LLM response is blank.

---

### `LLMRanker`

Scores and sorts candidate tasks against a query task using Pydantic-validated LLM output.

```python
from orcanet.retrieval import LLMRanker

ranker = LLMRanker(llm=my_llm)
ranked = await ranker.rerank(query_task, candidates, top_k=5)
# → [(task_a, 0.92, "same domain, similar n_classes"), ...]
```

**Constructor**

| Parameter | Type | Description |
|---|---|---|
| `llm` | `BaseLLM` | LangChain-compatible LLM; called via `ainvoke` |

**`async rerank(query_task, candidate_tasks, top_k=10) -> list[tuple[Task, float, str]]`**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query_task` | `Task` | — | The target task against which candidates are ranked |
| `candidate_tasks` | `list[Task]` | — | Candidate pool; returns `[]` immediately when empty (no LLM call) |
| `top_k` | `int` | `10` | Maximum number of results to return |

**Returns** a list of `(Task, score, reasoning)` tuples sorted by `score` descending. `score` is in `[0.0, 1.0]`. `reasoning` is the LLM's one-sentence explanation for the ranking position.

The LLM response is validated against `_RankedList(rankings: list[_RankedItem])` where `_RankedItem.score` carries `Field(ge=0.0, le=1.0)`. Any parse failure (`json.JSONDecodeError`, `ValidationError`) or out-of-range score returns `[]` with a `WARNING` log rather than raising.

---

### `HybridRetriever`

Three-stage async retrieval pipeline wiring FAISS, `TaskRepository`, `CrossDomainEmbedder`, `QueryExpander`, and `LLMRanker` into a single cohesive interface.

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
```

**Constructor parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `faiss_index` | object | — | Must expose `.search(embedding: np.ndarray, k: int) → list[tuple[str, float]]` |
| `task_repository` | `TaskRepository` | — | `async get_by_id(UUID) → Task \| None` |
| `embedder` | `CrossDomainEmbedder` | — | Maps the 25-dim feature vector to the FAISS embedding space |
| `query_expander` | `QueryExpander` | — | Used only by `retrieve_with_expanded_queries` |
| `llm_ranker` | `LLMRanker` | — | Activated in Stage 3 when conditions are met |
| `top_k_initial` | `int` | `50` | FAISS candidate count (Stage 1) |
| `top_k_final` | `int` | `10` | Maximum results returned |
| `similarity_threshold` | `float` | `0.6` | FAISS scores below this are discarded in Stage 2 |
| `use_llm_reranking` | `bool` | `True` | Set to `False` to skip Stage 3 entirely |

**`async retrieve(query_task, filters=None) -> list[tuple[Task, float, str]]`**

Executes the three-stage pipeline:

1. **Stage 1 — FAISS**: converts `query_task` to a 25-dim float32 feature vector (`log1p(n_samples)`, `n_features`, `n_classes`; `None` → 0), embeds via `CrossDomainEmbedder.embed`, searches the index.
2. **Stage 2 — Filter**: batch-fetches all candidate UUIDs concurrently via `asyncio.gather(..., return_exceptions=True)`. Failed individual fetches are logged at `WARNING` and skipped without aborting the batch. `None` results (unknown or deleted tasks) are dropped. Candidates below `similarity_threshold` are discarded. The optional `filters` dict applies field-equality checks.
3. **Stage 3 — LLM re-rank**: delegates to `LLMRanker.rerank` only when `use_llm_reranking=True` and `len(candidates) > top_k_final`; otherwise returns top-`top_k_final` candidates annotated as `"vector similarity"`.

**`async retrieve_with_expanded_queries(query_description, query_task) -> list[tuple[Task, float, str]]`**

Calls `QueryExpander.expand(query_description)`, then for the original description and each expansion creates a task variant via `query_task.model_copy(update={"name": description})` and calls `retrieve(query_variant)`. Each variant produces a distinct FAISS embedding, making the fan-out semantically meaningful rather than redundant. Results are merged via `_deduplicate_and_sort` — duplicate `task_id` entries are collapsed to the highest-scoring occurrence — and the top-`top_k_final` are returned.

---

## OrcaNet Reasoning Python API

The reasoning subpackage provides the LangChain-powered transfer recommendation agent as a pure-Python API. All public symbols are exported from `orcanet.reasoning`.

```python
from orcanet.reasoning import (
    OrcaNetAgent,
    TransferRecommendationResponse,
    SourceTaskRecommendation,
    LLMParsingError,
)
```

### `LLMParsingError`

Custom exception raised by `OrcaNetAgent.recommend_transfer` when the LLM's output cannot be parsed into a `TransferRecommendationResponse` after all retry attempts are exhausted.

```python
try:
    result = await agent.recommend_transfer(query)
except LLMParsingError as exc:
    print(f"All retries failed: {exc}")
```

`LLMParsingError` subclasses `Exception` directly. The message string includes the number of attempts made and the last raw LLM output.

### `SourceTaskRecommendation`

Pydantic v2 model for a single source-task entry in the agent's response.

| Field | Type | Constraints | Description |
|---|---|---|---|
| `task_id` | `str` | — | UUID string of the recommended source task |
| `task_name` | `str` | — | Human-readable source task name |
| `similarity_score` | `float` | `[0.0, 1.0]` | Embedding cosine similarity to the target task |
| `transfer_score` | `float` | `[0.0, 1.0]` | Transferability score from `transfer_scoring_tool` |
| `reasoning` | `str` | — | Agent's natural-language rationale |

### `TransferRecommendationResponse`

Pydantic v2 model for the full structured response returned by `OrcaNetAgent.recommend_transfer`.

| Field | Type | Constraints | Description |
|---|---|---|---|
| `top_sources` | `list[SourceTaskRecommendation]` | may be empty | Ranked list of recommended source tasks |
| `recommended_strategy` | `TransferStrategy` | `Literal["feature","weight","architecture","multi_task"]` | Transfer strategy name; Pydantic raises `ValidationError` for any value outside this set |
| `expected_improvement` | `float` | `[0.0, 1.0]` | Predicted relative improvement from applying the transfer |
| `explanation` | `str` | — | Free-text explanation |
| `confidence` | `float` | `[0.0, 1.0]` | Agent's self-assessed confidence in the recommendation |

`TransferStrategy` is a public type alias exported from `orcanet.reasoning`:

```python
from orcanet.reasoning import TransferStrategy  # Literal["feature","weight","architecture","multi_task"]
```

### `OrcaNetAgent`

LangChain-backed reasoning agent that runs a tool-augmented loop over four `@tool`-decorated async functions and returns a validated `TransferRecommendationResponse`.

#### Constructor

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
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `llm_provider` | `str` | `"openai"` | `"openai"` → `ChatOpenAI`; `"anthropic"` → `ChatAnthropic`; `"local"` → `ChatOpenAI` with `base_url` from `ORCANET_LOCAL_LLM_URL` env var (default `http://localhost:11434/v1`); any other value raises `ValueError` |
| `model` | `str` | `"gpt-4-turbo"` | Model name forwarded to the LLM constructor |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `api_key` | `str \| None` | `None` | API key for the chosen LLM provider |
| `retriever` | `HybridRetriever \| None` | `None` | Injected into `task_retrieval_tool` |
| `embedder` | `CrossDomainEmbedder \| None` | `None` | Injected into `embedding_similarity_tool` |
| `task_repository` | `TaskRepository \| None` | `None` | Injected into `embedding_similarity_tool`, `transfer_scoring_tool`, and `performance_prediction_tool` |
| `transfer_strategies` | `dict[str, TransferStrategy] \| None` | `None` | Injected into `transfer_scoring_tool`; keys are strategy names the LLM can request |
| `orcamind_client` | `OrcaMindClient \| None` | `None` | Injected into `performance_prediction_tool` |

Services not provided at construction time can be injected later by calling the module-level `set_*()` functions directly (e.g. `orcanet.reasoning.tools.task_retrieval_tool.set_retriever(r)`).

#### `async recommend_transfer(query: str) -> TransferRecommendationResponse`

Runs the agent loop for the given natural-language query and returns a validated `TransferRecommendationResponse`.

**Retry behaviour:** up to `_MAX_RETRIES = 2` additional attempts (3 total) are made when the LLM's final message cannot be parsed. Each retry appends a corrective suffix to the original query instructing the LLM to respond only with valid JSON matching the required schema. `LLMParsingError` is raised after all attempts are exhausted.

| Attempt | Outcome | Next action |
|---|---|---|
| 1 | Parse succeeds | Return `TransferRecommendationResponse` |
| 1 | Parse fails | Log warning; retry with `query + corrective_suffix` |
| 2 | Parse succeeds | Return immediately |
| 2 | Parse fails | Log warning; retry again |
| 3 | Parse succeeds | Return immediately |
| 3 | Parse fails | Raise `LLMParsingError` |

#### Tool functions

The agent is equipped with four tools. Each tool's docstring is used by LangChain as the tool description shown to the LLM. All tools return JSON-encoded strings and return `{"error": "..."}` on configuration errors or lookup failures rather than raising.

`OrcaNetAgent` uses per-instance `StructuredTool` objects produced by `make_<tool_name>(*deps)` factory functions. The factories close over the supplied dependencies, so each agent instance is fully isolated and constructing two agents with different dependencies does not cause cross-instance state leakage. The module-level `@tool`-decorated functions remain available for direct standalone use.

| Tool | Signature | Returns | Notes |
|---|---|---|---|
| `task_retrieval_tool` | `(query: str, filters: str = "{}") → str` | JSON array of task objects with `task_id`, `score`, and `reason` fields | Factory: `make_task_retrieval_tool(retriever)` |
| `embedding_similarity_tool` | `(task_id_a: str, task_id_b: str) → str` | `{"similarity": float}` in `[-1.0, 1.0]`; both embeddings are L2-normalised before the dot product | Factory: `make_embedding_similarity_tool(embedder, task_repository)` |
| `transfer_scoring_tool` | `(source_task_id: str, target_task_id: str, strategy: str = "feature") → str` | `{"overall": float, "layer_scores": {...}, "recommended_layers": [...], "reasoning": "..."}` | Factory: `make_transfer_scoring_tool(transfer_strategies, task_repository)` |
| `performance_prediction_tool` | `(task_id: str, model_config_json: str) → str` | `{"metrics": {**final_metrics}, "experiment_id": "..."}` — `model_config_json` must be a JSON object; arrays, scalars, or strings return an error | Factory: `make_performance_prediction_tool(orcamind_client, task_repository)` |

#### End-to-end example

```python
import asyncio
from orcanet.reasoning import OrcaNetAgent

async def main():
    agent = OrcaNetAgent(
        llm_provider="openai",
        model="gpt-4-turbo",
        api_key="sk-...",
        retriever=hybrid_retriever,
        embedder=cross_domain_embedder,
        task_repository=task_repo,
        transfer_strategies={"feature": feature_strategy},
        orcamind_client=orcamind_client,
    )

    result = await agent.recommend_transfer(
        "Find the best source task for fine-tuning a retinal scan classifier."
    )

    print(result.recommended_strategy)          # "feature"
    print(result.confidence)                     # 0.82
    print(result.expected_improvement)           # 0.18
    print(result.top_sources[0].task_name)       # "brain MRI classification"
    print(result.top_sources[0].similarity_score) # 0.88
    print(result.explanation)                    # full LLM explanation

asyncio.run(main())
```

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
| `transfer.py`      | `TransferMapping`, `TransferScore`, `TransferRecommendation` |

