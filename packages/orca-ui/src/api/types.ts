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
