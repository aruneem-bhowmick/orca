/**
 * TypeScript interfaces mirroring the Orca Web BFF Pydantic schemas.
 * These types are used throughout the frontend for type-safe API
 * communication and state management.
 *
 * @module api/types
 */

/** Authenticated user profile returned by `GET /auth/me`. */
export interface User {
  user_id: string;
  email: string;
  username: string;
  role: string;
  preferences: Record<string, unknown> | null;
}

/** JWT access token response from login, register, and refresh endpoints. */
export interface TokenResponse {
  access_token: string;
  token_type: string;
}

/** Request body for `POST /auth/register`. */
export interface RegisterRequest {
  email: string;
  username: string;
  password: string;
}

/** Request body for `POST /auth/login`. */
export interface LoginRequest {
  email: string;
  password: string;
}

/** Request body for `PATCH /auth/me` to update the user profile. */
export interface ProfileUpdate {
  username?: string;
  preferences?: Record<string, unknown>;
}

/**
 * Generic paginated response envelope used by history, bookmark,
 * and feed endpoints.
 *
 * @typeParam T - The type of items in the paginated list.
 */
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  per_page: number;
  pages: number;
}

/** A single entry in the user's activity log. */
export interface ActivityLogEntry {
  id: string;
  user_id: string;
  action: string;
  resource_type: string | null;
  resource_id: string | null;
  service: string | null;
  details: Record<string, unknown> | null;
  created_at: string;
}

/** A user bookmark referencing a task, experiment, or other resource. */
export interface Bookmark {
  id: string;
  user_id: string;
  resource_type: string;
  resource_id: string;
  note: string | null;
  created_at: string;
}

/** Request body for `POST /bookmarks`. */
export interface BookmarkCreateRequest {
  resource_type: string;
  resource_id: string;
  note?: string;
}

/** Health check response from `GET /health`. */
export interface HealthStatus {
  status: "healthy" | "degraded";
  services: {
    postgres: boolean;
    redis: boolean;
    orcamind: boolean;
    orcalab: boolean;
    orcanet: boolean;
  };
}

/**
 * Public platform statistics returned by `GET /dashboard/stats`.
 * Used on the landing page to display live counters.
 */
export interface DashboardStats {
  tasks_registered: number;
  experiments_run: number;
  transfers_scored: number;
}

/**
 * Aggregated platform statistics returned by `GET /dashboard/overview`.
 * Used on the main dashboard page to populate the summary stat cards.
 */
export interface DashboardOverview {
  total_tasks: number;
  running_experiments: number;
  completed_experiments: number;
  recent_transfers: number;
}

/** An OrcaMind task record returned by `GET /orcamind/tasks/:id`. */
export interface Task {
  task_id: string;
  name: string;
  domain: string;
  task_type: string;
  n_samples: number;
  n_features: number | null;
  n_classes: number | null;
  metadata: Record<string, unknown> | null;
  created_at: string;
}

/** A model recommendation entry from `POST /orcamind/recommend`. */
export interface ModelRecommendation {
  model_id: string;
  model_name: string;
  architecture: string;
  predicted_accuracy: number;
  confidence: number;
  config: Record<string, unknown> | null;
}

/** A similar-task entry from `POST /orcamind/similar-tasks`. */
export interface SimilarTask {
  task_id: string;
  name: string;
  domain: string;
  task_type: string;
  similarity_score: number;
}

/** A performance-prediction result from `POST /orcamind/predict-performance`. */
export interface PerformancePrediction {
  predicted_accuracy: number;
  confidence: number;
  model_id: string;
}

/** Request body for `POST /orcamind/recommend`. */
export interface RecommendRequest {
  task_id: string;
  top_k?: number;
}

/** Request body for `POST /orcamind/similar-tasks`. */
export interface SimilarTasksRequest {
  task_id: string;
  top_k?: number;
}

/** Request body for `POST /orcamind/predict-performance`. */
export interface PredictPerformanceRequest {
  task_id: string;
  model_config: Record<string, unknown>;
}

/** Request body for `POST /orcamind/tasks` (embed a new task into OrcaMind). */
export interface EmbedTaskRequest {
  name: string;
  domain: string;
  task_type: string;
  n_samples: number;
  n_features?: number;
  n_classes?: number;
  metadata?: Record<string, unknown>;
}

/**
 * Possible lifecycle states for an OrcaLab experiment or sweep.
 * `pending` – not yet started; `running` – actively training;
 * `completed` – finished successfully; `failed` – terminated with an error.
 */
export type ExperimentStatus = "pending" | "running" | "completed" | "failed";

/**
 * An OrcaLab experiment record returned by
 * `GET /orcalab/experiments` and `GET /orcalab/experiments/:id`.
 */
export interface Experiment {
  experiment_id: string;
  name: string;
  task_id: string;
  model_id: string;
  status: ExperimentStatus;
  started_at: string | null;
  completed_at: string | null;
  training_config: Record<string, unknown> | null;
  metrics: Record<string, number> | null;
  mlflow_run_id: string | null;
  created_at: string;
}

/** Request body for `POST /orcalab/experiments`. */
export interface CreateExperimentRequest {
  task_id: string;
  model_id: string;
  training_config?: Record<string, unknown>;
}

/**
 * A single trial inside a hyperparameter sweep, as returned in
 * `Sweep.results`.
 */
export interface SweepTrial {
  /** Sequential trial index (1-based). */
  trial_id: number;
  /** Hyperparameter values sampled for this trial. */
  params: Record<string, number | string | boolean>;
  /** Metric values recorded at trial completion. */
  metrics: Record<string, number>;
}

/**
 * An OrcaLab hyperparameter sweep record returned by
 * `GET /orcalab/sweeps` and `GET /orcalab/sweeps/:id`.
 */
export interface Sweep {
  sweep_id: string;
  task_id: string;
  search_strategy: string;
  n_trials: number;
  completed_trials: number;
  status: ExperimentStatus;
  best_trial: number | null;
  results: SweepTrial[] | null;
  created_at: string;
}

/** Request body for `POST /orcalab/sweeps`. */
export interface CreateSweepRequest {
  task_id: string;
  search_strategy: string;
  n_trials: number;
  use_orcamind_priors?: boolean;
}

/**
 * A real-time metric update delivered over the WebSocket connection
 * at `WS /orcalab/ws/experiments/:id/live`.
 */
export interface MetricUpdate {
  /** Current training epoch (1-based). */
  epoch: number;
  /** Training loss at this epoch. */
  loss: number;
  /** Validation accuracy at this epoch, if available. */
  accuracy: number | null;
  /** Effective learning rate at this epoch, if available. */
  learning_rate: number | null;
  /** ISO-8601 timestamp from the training process. */
  timestamp: string;
}

/**
 * A WebSocket control message sent from the browser to the BFF to
 * pause, resume, or cancel an in-progress experiment.
 */
export interface ExperimentControl {
  action: "pause" | "resume" | "cancel";
}

// ---------------------------------------------------------------------------
// OrcaNet types
// ---------------------------------------------------------------------------

/** Request body for `POST /orcanet/transfer/score`. */
export interface TransferScoreRequest {
  source_task_id: string;
  target_task_id: string;
}

/**
 * Transfer score response from `POST /orcanet/transfer/score`.
 * The `score` field is a value in [0, 1] where higher means more
 * transferable knowledge.
 */
export interface TransferScoreResult {
  score: number;
  source_task_id: string;
  target_task_id: string;
}

/** Request body for `POST /orcanet/transfer/recommend`. */
export interface TransferRecommendRequest {
  target_task_id: string;
  top_k?: number;
}

/**
 * A single ranked transfer recommendation returned by
 * `POST /orcanet/transfer/recommend`.
 */
export interface TransferRecommendation {
  /** ID of the source task providing the transfer knowledge. */
  source_task_id: string;
  /** Human-readable name of the source task. */
  source_task_name: string;
  /** Predicted transferability score in [0, 1]. */
  transfer_score: number;
  /** Recommended transfer strategy (e.g. "fine-tune", "feature-extract"). */
  strategy: string;
  /** Optional hyperparameter config for the recommended strategy. */
  config: Record<string, unknown> | null;
}

/** Request body for `POST /orcanet/retrieve`. */
export interface RetrieveRequest {
  query: string;
  top_k?: number;
}

/**
 * A single search result from `POST /orcanet/retrieve`, representing an
 * OrcaMind task that semantically matches the natural-language query.
 */
export interface RetrieveResult {
  task_id: string;
  name: string;
  domain: string;
  task_type: string;
  similarity_score: number;
}

/** Request body for `POST /orcanet/explain`. */
export interface ExplainRequest {
  source_task_id: string;
  target_task_id: string;
  strategy?: string;
}

/**
 * LLM-generated explanation returned by `POST /orcanet/explain` describing
 * why the recommended transfer strategy is suitable for the given task pair.
 */
export interface ExplainResult {
  explanation: string;
  source_task_id: string;
  target_task_id: string;
}
