/**
 * Mock data fixtures for orca-ui tests.
 *
 * Provides typed mock objects matching the BFF's response schemas.
 * Used across all test suites to ensure consistent test data.
 *
 * @module test/mocks/handlers
 */
import type {
  User,
  TokenResponse,
  HealthStatus,
  DashboardStats,
  DashboardOverview,
  Task,
  ModelRecommendation,
  SimilarTask,
  PerformancePrediction,
  ActivityLogEntry,
  PaginatedResponse,
} from "@/api/types";

/** Mock authenticated user profile. */
export const mockUser: User = {
  user_id: "550e8400-e29b-41d4-a716-446655440000",
  email: "test@example.com",
  username: "testuser",
  role: "user",
  preferences: null,
};

/** Mock JWT token response from login/register/refresh endpoints. */
export const mockTokenResponse: TokenResponse = {
  access_token: "mock-access-token-jwt",
  token_type: "bearer",
};

/** Mock health check response with all services healthy. */
export const mockHealthStatus: HealthStatus = {
  status: "healthy",
  services: {
    postgres: true,
    redis: true,
    orcamind: true,
    orcalab: true,
    orcanet: true,
  },
};

/** Mock dashboard statistics for the landing page live counters. */
export const mockDashboardStats: DashboardStats = {
  tasks_registered: 42,
  experiments_run: 128,
  transfers_scored: 56,
};

/** Mock aggregated overview stats returned by GET /dashboard/overview. */
export const mockDashboardOverview: DashboardOverview = {
  total_tasks: 12,
  running_experiments: 3,
  completed_experiments: 27,
  recent_transfers: 5,
};

/** Mock OrcaMind task record. */
export const mockTask: Task = {
  task_id: "task-uuid-001",
  name: "Image Classification",
  domain: "vision",
  task_type: "classification",
  n_samples: 50000,
  n_features: 2048,
  n_classes: 10,
  metadata: null,
  created_at: "2024-03-15T10:00:00Z",
};

/** Mock list of two OrcaMind tasks for list view tests. */
export const mockTaskList: Task[] = [
  mockTask,
  {
    task_id: "task-uuid-002",
    name: "Sentiment Analysis",
    domain: "nlp",
    task_type: "classification",
    n_samples: 25000,
    n_features: null,
    n_classes: 3,
    metadata: null,
    created_at: "2024-03-20T08:00:00Z",
  },
];

/** Mock model recommendations from POST /orcamind/recommend. */
export const mockRecommendations: ModelRecommendation[] = [
  {
    model_id: "model-001",
    model_name: "ResNet-50",
    architecture: "ResNet",
    predicted_accuracy: 0.923,
    confidence: 0.87,
    config: null,
  },
  {
    model_id: "model-002",
    model_name: "EfficientNet-B3",
    architecture: "EfficientNet",
    predicted_accuracy: 0.911,
    confidence: 0.81,
    config: null,
  },
];

/** Mock similar task results from POST /orcamind/similar-tasks. */
export const mockSimilarTasks: SimilarTask[] = [
  {
    task_id: "task-uuid-003",
    name: "Object Detection",
    domain: "vision",
    task_type: "detection",
    similarity_score: 0.94,
  },
  {
    task_id: "task-uuid-004",
    name: "Face Recognition",
    domain: "vision",
    task_type: "classification",
    similarity_score: 0.87,
  },
];

/** Mock performance prediction result from POST /orcamind/predict-performance. */
export const mockPerformancePrediction: PerformancePrediction = {
  predicted_accuracy: 0.897,
  confidence: 0.76,
  model_id: "model-001",
};

/** Mock activity log entry. */
export const mockActivityEntry: ActivityLogEntry = {
  id: "log-001",
  user_id: "550e8400-e29b-41d4-a716-446655440000",
  action: "task_created",
  resource_type: "task",
  resource_id: "task-uuid-001",
  service: "orcamind",
  details: null,
  created_at: "2024-03-15T11:00:00Z",
};

/** Mock paginated activity log response with a single entry. */
export const mockActivityPage: PaginatedResponse<ActivityLogEntry> = {
  items: [mockActivityEntry],
  total: 1,
  page: 1,
  per_page: 10,
  pages: 1,
};
