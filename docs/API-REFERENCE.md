# API Reference

> Part of the [Orca](../README.md) meta-learning platform.

---

## Overview

The Orca platform exposes three independent HTTP services:

| Service   | Default Port | Base URL              | Description                                        |
|-----------|-------------|-----------------------|----------------------------------------------------|
| OrcaMind  | `8000`       | `http://localhost:8000` | Meta-learning engine and recommendations           |
| OrcaLab   | `8001`       | `http://localhost:8001` | Experiment orchestration and search                |
| OrcaNet   | `8002`       | `http://localhost:8002` | Cross-domain knowledge transfer agent              |

All services auto-generate interactive API docs at `GET /docs` (Swagger UI) and `GET /redoc`.

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

## OrcaNet API — port 8002

> The OrcaNet API is currently scaffolded. Endpoint implementations are in progress; this section describes the planned interface.

OrcaNet orchestrates OrcaMind and OrcaLab to deliver end-to-end cross-domain knowledge transfer. It retrieves the best-performing model config for a source task from OrcaMind, scores transferability via Centered Kernel Alignment (CKA), dispatches a validation experiment to OrcaLab when the transfer score exceeds the threshold, and returns a structured recommendation with an LLM-generated explanation.

### Transfer

#### `POST /api/v1/transfer` — Recommend a transfer

Status: **202 Accepted**

Scores transferability from all candidate source tasks to the given target task, optionally validates the best candidate via OrcaLab, and returns the top recommendation with a reasoning trace.

**Request body** — `TransferRequest`

```json
{
  "target_task_id":      "uuid",
  "top_k":               5,
  "min_transfer_score":  0.4,
  "run_validation":      true
}
```

| Field                | Type      | Default | Description |
|----------------------|-----------|---------|-------------|
| `target_task_id`     | `string`  | —       | UUID of the target task already registered in OrcaMind |
| `top_k`              | `int`     | `5`     | Number of source candidates to retrieve from the hybrid retrieval stage |
| `min_transfer_score` | `float`   | `0.4`   | Candidates with CKA score below this are not dispatched to OrcaLab for validation |
| `run_validation`     | `bool`    | `true`  | When `false`, returns the recommendation without waiting for an OrcaLab experiment |

**Response** — `TransferRecommendation`

```json
{
  "recommendation_id":  "uuid",
  "target_task_id":     "uuid",
  "source_task_id":     "uuid",
  "transfer_score":     0.72,
  "model_config":       { "architecture": "resnet18", "config": {} },
  "validation_result":  { "experiment_id": "uuid", "metrics": {} },
  "explanation":        "string",
  "status":             "completed | pending | failed"
}
```

---

#### `GET /api/v1/transfer/{recommendation_id}` — Get transfer status

**Response** — `TransferRecommendation` or **404**

Poll this endpoint after a `POST /api/v1/transfer` with `run_validation: true` to wait for the OrcaLab validation experiment to complete.

---

### Retrieval

#### `POST /api/v1/similar-tasks` — Find similar tasks

Three-stage hybrid retrieval: FAISS vector similarity → PostgreSQL metadata filter → optional LLM re-ranking.

**Request body** — `SimilarTasksRequest`

```json
{
  "target_task_id": "uuid",
  "top_k":          10,
  "use_reranking":  true
}
```

**Response** — `list[SimilarTaskResult]`

```json
[
  { "task_id": "uuid", "score": 0.91, "rank": 1, "reason": "string | null" }
]
```

---

### Architecture

#### `POST /api/v1/recommend-architecture` — Recommend architecture for target task

Fetches model candidates from OrcaMind, applies domain-adversarial embeddings to score cross-domain transferability, and returns ranked architectures.

**Request body**

```json
{
  "target_task_id": "uuid",
  "top_k":          5
}
```

**Response** — `list[ModelRecommendation]` (same schema as OrcaMind `/recommend-model`)

---

### Reasoning

#### `POST /api/v1/explain-transfer` — Generate transfer explanation

Runs the LangChain ReAct agent to produce a human-readable explanation of why a particular source–target pair is a strong transfer candidate.

**Request body**

```json
{
  "source_task_id": "uuid",
  "target_task_id": "uuid",
  "transfer_score":  0.72
}
```

**Response**

```json
{ "explanation": "string", "reasoning_trace": ["string"] }
```

---

### Transfer Mappings

#### `GET /api/v1/transfer-mappings` — List stored transfer mappings

Returns persisted pairwise source→target transfer scores from the `transfer_mappings` registry table.

**Query parameters**: `limit` (default 50, max 500), `offset` (default 0), `source_task_id` (filter)

**Response** — `list[TransferMapping]`

---

## Health Endpoints

All three services expose a health endpoint with no authentication requirement.

| Service  | Endpoint           | Healthy response                                                              |
|----------|--------------------|-------------------------------------------------------------------------------|
| OrcaMind | `GET /health`      | `{"status": "ok", "faiss": true \| false}`                                     |
| OrcaLab  | `GET /health`      | `{"status": "ok", "prefect": "http://..." \| null}`                            |
| OrcaNet  | `GET /health`      | `{"status": "ok", "orcamind": "http://...", "orcalab": "http://..."}`          |

The OrcaMind health endpoint reports `faiss: false` when the FAISS index has not been built yet. The service remains fully operational — only `/recommend-model` and `/similar-tasks` return 503 until the index is populated.

---

## OrcaNet Transfer Python API

The transfer subpackage provides a pure-Python API for two concrete transfer strategies: CKA-based feature-level scoring (`FeatureTransfer`) and direct parameter-tensor transfer with selective layer matching (`WeightTransfer`). Neither strategy exposes an HTTP endpoint in the current release — they are consumed directly by OrcaNet's internal recommendation pipeline.

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
| `overall` | `float` | Aggregate transferability score in [0, 1]. `FeatureTransfer`: depth-weighted mean CKA. `WeightTransfer`: `n_matched / n_total` parameter ratio. |
| `layer_scores` | `dict[str, float]` | Per-named-layer score. `FeatureTransfer`: CKA similarity value. `WeightTransfer`: `1.0` (matched) or `0.0` (unmatched) per parameter tensor. |
| `recommended_layers` | `list[str]` | Layers selected for transfer. `FeatureTransfer`: layers whose CKA exceeds `cka_threshold`. `WeightTransfer`: all matched parameter names. |
| `reasoning` | `str` | Human-readable summary, e.g. `"CKA analysis: 3/4 layers exceed threshold 0.5."` or `"Matched 4/4 layers by name"`. |

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

**`execute_transfer(source: Task, target: Task, source_model: nn.Module) -> tuple[nn.Module, list[str]]`**

Deep-copy `source_model` to create the base target model, then for each parameter tensor:

- If a safe source tensor can be resolved (name match + shape agreement, or shape-first scan for `match_by="shape"`): copy it in-place via `tensor.copy_()`.
- Otherwise: reinitialise with `kaiming_uniform_` (2-D+ tensors, e.g. weight matrices) or `zeros_` (1-D tensors, e.g. bias vectors).

Returns `(adapted_model, transferred_names)`. Pass `transferred_names` directly to `get_optimizer_with_layer_lr`.

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

adapted, transferred = transfer.execute_transfer(source_task, target_task, source_model)
# adapted: nn.Module — source architecture with unmatched params reinitialised
# transferred: list[str] — e.g. ['0.weight', '0.bias', '2.weight', '2.bias']

optimizer = get_optimizer_with_layer_lr(
    adapted,
    transferred_layers=transferred,
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
| `transferred_layers` | Parameter names (as returned by `model.named_parameters()`) that were copied from the source. Typically the second element of the `execute_transfer` return tuple. |
| `base_lr` | Learning rate for non-transferred (new / reinitialised) parameters. |
| `decay` | Multiplicative factor applied to `base_lr` for transferred parameters. Default `0.1`. |

**Returns** `torch.optim.Adam` with one parameter group per `named_parameters()` entry.

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

