import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery, useMutation } from "@tanstack/react-query";
import apiClient from "@/api/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { formatDate } from "@/lib/utils";
import { ROUTES } from "@/lib/constants";
import { RecommendationCards } from "./Recommendations";
import type {
  Task,
  ModelRecommendation,
  SimilarTask,
  PerformancePrediction,
  Bookmark,
} from "@/api/types";

/** Preset model architectures available for performance prediction. */
const MODEL_OPTIONS = [
  "ResNet",
  "DistilBERT",
  "XGBoost",
  "SVM",
  "MLP",
] as const;

type ModelOption = (typeof MODEL_OPTIONS)[number];

/**
 * Inline section that fetches and displays similar tasks from OrcaMind.
 *
 * @param props.taskId - The source task ID to find neighbors for.
 */
function SimilarTasksSection({ taskId }: { taskId: string }) {
  const navigate = useNavigate();
  const {
    mutate: findSimilar,
    data: similarTasks,
    isPending,
    error,
  } = useMutation({
    mutationFn: async () => {
      const res = await apiClient.post<SimilarTask[]>(
        "/orcamind/similar-tasks",
        { task_id: taskId },
      );
      return res.data;
    },
  });

  return (
    <div>
      <div className="flex items-center gap-3">
        <h3 className="text-base font-semibold">Similar Tasks</h3>
        <Button
          variant="outline"
          size="sm"
          onClick={() => findSimilar()}
          disabled={isPending}
          data-testid="find-similar-btn"
        >
          {isPending ? "Searching…" : "Find Similar Tasks"}
        </Button>
      </div>
      {error && (
        <p className="mt-2 text-sm text-destructive" data-testid="similar-error">
          Failed to find similar tasks.
        </p>
      )}
      {similarTasks && similarTasks.length === 0 && (
        <p className="mt-2 text-sm text-muted-foreground">
          No similar tasks found.
        </p>
      )}
      {similarTasks && similarTasks.length > 0 && (
        <div className="mt-3 overflow-auto rounded-lg border" data-testid="similar-table">
          <table className="w-full text-sm">
            <thead className="border-b bg-muted/50 text-left">
              <tr>
                <th className="px-4 py-2 font-medium">Name</th>
                <th className="px-4 py-2 font-medium">Domain</th>
                <th className="px-4 py-2 font-medium">Type</th>
                <th className="px-4 py-2 font-medium">Similarity</th>
              </tr>
            </thead>
            <tbody>
              {similarTasks.map((t) => (
                <tr
                  key={t.task_id}
                  className="cursor-pointer border-b last:border-0 hover:bg-muted/50"
                  onClick={() =>
                    navigate(`${ROUTES.ORCAMIND_TASKS}/${t.task_id}`)
                  }
                  data-testid={`similar-row-${t.task_id}`}
                >
                  <td className="px-4 py-2 font-medium">{t.name}</td>
                  <td className="px-4 py-2 text-muted-foreground">{t.domain}</td>
                  <td className="px-4 py-2">{t.task_type}</td>
                  <td className="px-4 py-2">
                    {(t.similarity_score * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/**
 * Inline section for predicting model performance on this task.
 *
 * The user selects a model architecture from a preset list and triggers
 * a performance prediction request to `POST /orcamind/predict-performance`.
 *
 * @param props.taskId - The task ID to predict performance for.
 */
function PredictPerformanceSection({ taskId }: { taskId: string }) {
  const [selectedModel, setSelectedModel] = useState<ModelOption>("ResNet");

  const {
    mutate: predict,
    data: prediction,
    isPending,
    error,
  } = useMutation({
    mutationFn: async () => {
      const res = await apiClient.post<PerformancePrediction>(
        "/orcamind/predict-performance",
        {
          task_id: taskId,
          model_config: { model_name: selectedModel },
        },
      );
      return res.data;
    },
  });

  return (
    <div>
      <h3 className="text-base font-semibold">Predict Performance</h3>
      <div className="mt-3 flex items-center gap-3">
        <label htmlFor="model-select" className="text-sm font-medium">
          Model
        </label>
        <select
          id="model-select"
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value as ModelOption)}
          className="rounded-md border border-input bg-background px-3 py-1.5 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          data-testid="model-select"
        >
          {MODEL_OPTIONS.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
        <Button
          variant="outline"
          size="sm"
          onClick={() => predict()}
          disabled={isPending}
          data-testid="predict-btn"
        >
          {isPending ? "Predicting…" : "Predict Performance"}
        </Button>
      </div>
      {error && (
        <p className="mt-2 text-sm text-destructive" data-testid="predict-error">
          Failed to predict performance.
        </p>
      )}
      {prediction && (
        <div
          className="mt-3 flex gap-6 rounded-lg border p-4"
          data-testid="prediction-result"
        >
          <div>
            <p className="text-xs text-muted-foreground">Predicted accuracy</p>
            <p className="text-2xl font-bold">
              {(prediction.predicted_accuracy * 100).toFixed(1)}%
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Confidence</p>
            <p className="text-2xl font-bold">
              {(prediction.confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * OrcaMind task detail page.
 *
 * Fetches a single task by ID from `GET /orcamind/tasks/:id` and displays
 * its metadata. Provides three interactive sections:
 * - **Recommendations** – calls `POST /orcamind/recommend` and renders ranked
 *   model recommendation cards via `RecommendationCards`.
 * - **Similar Tasks** – calls `POST /orcamind/similar-tasks` and shows a
 *   clickable similarity table.
 * - **Predict Performance** – lets the user pick a model architecture and
 *   calls `POST /orcamind/predict-performance` to show predicted accuracy and
 *   confidence.
 *
 * A bookmark toggle at the top of the page calls `POST /bookmarks` or
 * `DELETE /bookmarks/:id` to save or remove the task from the user's
 * bookmark list.
 */
export function TaskDetail() {
  const { id: taskId } = useParams<{ id: string }>();
  const [bookmarked, setBookmarked] = useState(false);
  const [bookmarkId, setBookmarkId] = useState<string | null>(null);

  const {
    data: task,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["orcamind-task", taskId],
    queryFn: async () => {
      const res = await apiClient.get<Task>(`/orcamind/tasks/${taskId}`);
      return res.data;
    },
    enabled: !!taskId,
    staleTime: 60_000,
  });

  const {
    mutate: fetchRecommendations,
    data: recommendations,
    isPending: isLoadingRec,
    error: recError,
  } = useMutation({
    mutationFn: async () => {
      const res = await apiClient.post<ModelRecommendation[]>(
        "/orcamind/recommend",
        { task_id: taskId },
      );
      return res.data;
    },
  });

  async function handleBookmarkToggle() {
    if (bookmarked && bookmarkId) {
      try {
        await apiClient.delete(`/bookmarks/${bookmarkId}`);
        setBookmarked(false);
        setBookmarkId(null);
      } catch {
        /* ignore – bookmark may already be gone */
      }
    } else {
      try {
        const res = await apiClient.post<Bookmark>("/bookmarks", {
          resource_type: "task",
          resource_id: taskId,
        });
        setBookmarked(true);
        setBookmarkId(res.data.id);
      } catch {
        /* ignore – may already be bookmarked */
      }
    }
  }

  return (
    <div className="space-y-8">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold">Task Detail</h1>
          <p className="mt-1 text-muted-foreground">
            OrcaMind task metadata and analysis tools.
          </p>
        </div>
        <Button
          variant={bookmarked ? "secondary" : "outline"}
          size="sm"
          onClick={handleBookmarkToggle}
          data-testid="bookmark-btn"
        >
          {bookmarked ? "Bookmarked ✓" : "Bookmark"}
        </Button>
      </div>

      {/* Task metadata card */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Metadata</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <p className="text-muted-foreground" data-testid="task-loading">
              Loading task…
            </p>
          ) : isError || !task ? (
            <p className="text-destructive" data-testid="task-error">
              Failed to load task.
            </p>
          ) : (
            <dl
              className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-3"
              data-testid="task-metadata"
            >
              <div>
                <dt className="text-muted-foreground">Name</dt>
                <dd className="font-medium">{task.name}</dd>
              </div>
              <div>
                <dt className="text-muted-foreground">Domain</dt>
                <dd>{task.domain}</dd>
              </div>
              <div>
                <dt className="text-muted-foreground">Type</dt>
                <dd>{task.task_type}</dd>
              </div>
              <div>
                <dt className="text-muted-foreground">Samples</dt>
                <dd>{task.n_samples.toLocaleString()}</dd>
              </div>
              {task.n_features !== null && (
                <div>
                  <dt className="text-muted-foreground">Features</dt>
                  <dd>{task.n_features}</dd>
                </div>
              )}
              {task.n_classes !== null && (
                <div>
                  <dt className="text-muted-foreground">Classes</dt>
                  <dd>{task.n_classes}</dd>
                </div>
              )}
              <div>
                <dt className="text-muted-foreground">Created</dt>
                <dd>{formatDate(task.created_at)}</dd>
              </div>
            </dl>
          )}
        </CardContent>
      </Card>

      {/* Recommendations section */}
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold">Model Recommendations</h2>
          <Button
            variant="outline"
            size="sm"
            onClick={() => fetchRecommendations()}
            disabled={isLoadingRec}
            data-testid="get-recommendations-btn"
          >
            {isLoadingRec ? "Loading…" : "Get Recommendations"}
          </Button>
        </div>
        {recError && (
          <p className="text-sm text-destructive" data-testid="rec-error">
            Failed to load recommendations.
          </p>
        )}
        {recommendations && (
          <RecommendationCards
            recommendations={recommendations}
            onStartExperiment={() => {}}
          />
        )}
      </div>

      {/* Similar tasks section */}
      {taskId && <SimilarTasksSection taskId={taskId} />}

      {/* Predict performance section */}
      {taskId && <PredictPerformanceSection taskId={taskId} />}
    </div>
  );
}
