import { useQuery } from "@tanstack/react-query";
import { useSearchParams, useNavigate, Link } from "react-router-dom";
import apiClient from "@/api/client";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { ROUTES } from "@/lib/constants";
import type { ModelRecommendation } from "@/api/types";

/**
 * Display a ranked list of model recommendation cards.
 *
 * Each card shows the model name, architecture, predicted accuracy, and
 * confidence score. An optional "Start Experiment" action pre-navigates
 * to the OrcaLab experiments page so the user can initiate a training run
 * with the recommended configuration.
 *
 * @param props.recommendations - Ordered list of model recommendations.
 * @param props.onStartExperiment - Optional callback invoked when the user
 *   clicks "Start Experiment" on a card. If omitted, the button navigates
 *   to the OrcaLab experiments list.
 */
export function RecommendationCards({
  recommendations,
  onStartExperiment,
}: {
  recommendations: ModelRecommendation[];
  onStartExperiment?: (rec: ModelRecommendation) => void;
}) {
  const navigate = useNavigate();

  if (recommendations.length === 0) {
    return (
      <p className="text-muted-foreground" data-testid="no-recommendations">
        No recommendations available for this task.
      </p>
    );
  }

  return (
    <div
      className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3"
      data-testid="recommendation-cards"
    >
      {recommendations.map((rec, idx) => (
        <Card key={rec.model_id} data-testid={`rec-card-${rec.model_id}`}>
          <CardHeader>
            <div className="flex items-start justify-between">
              <CardTitle className="text-base">{rec.model_name}</CardTitle>
              <span className="rounded-full bg-secondary px-2 py-0.5 text-xs font-medium">
                #{idx + 1}
              </span>
            </div>
            <p className="text-sm text-muted-foreground">{rec.architecture}</p>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Predicted accuracy</span>
              <span className="font-medium">
                {(rec.predicted_accuracy * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Confidence</span>
              <span className="font-medium">
                {(rec.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <Button
              variant="secondary"
              size="sm"
              className="mt-3 w-full"
              onClick={() =>
                onStartExperiment
                  ? onStartExperiment(rec)
                  : navigate(ROUTES.ORCALAB_EXPERIMENTS)
              }
              data-testid={`start-experiment-${rec.model_id}`}
            >
              Start Experiment
            </Button>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

/**
 * Standalone OrcaMind recommendations page.
 *
 * Reads a `task_id` query parameter from the URL and fetches model
 * recommendations for that task via `POST /orcamind/recommend`. When no
 * `task_id` is present, renders a prompt directing the user to select a
 * task from the task list first.
 *
 * The query is disabled until `task_id` is known, so no network request is
 * made when the page is visited without a parameter.
 */
export function Recommendations() {
  const [searchParams] = useSearchParams();
  const taskId = searchParams.get("task_id");

  const { data: recommendations, isLoading, isError } = useQuery({
    queryKey: ["orcamind-recommendations", taskId],
    queryFn: async () => {
      const res = await apiClient.post<ModelRecommendation[]>(
        "/orcamind/recommend",
        { task_id: taskId },
      );
      return res.data;
    },
    enabled: !!taskId,
    staleTime: 60_000,
  });

  return (
    <div>
      <h1 className="text-2xl font-bold">Recommendations</h1>
      <p className="mt-1 text-muted-foreground">
        Model recommendations ranked by predicted task performance.
      </p>

      <div className="mt-6">
        {!taskId ? (
          <p className="text-muted-foreground" data-testid="no-task-selected">
            Select a task from the{" "}
            <Link
              to={ROUTES.ORCAMIND_TASKS}
              className="underline hover:text-foreground"
            >
              task list
            </Link>{" "}
            to view recommendations.
          </p>
        ) : isLoading ? (
          <p className="text-muted-foreground" data-testid="rec-loading">
            Loading recommendations…
          </p>
        ) : isError ? (
          <p className="text-destructive" data-testid="rec-error">
            Failed to load recommendations.
          </p>
        ) : (
          <RecommendationCards recommendations={recommendations ?? []} />
        )}
      </div>
    </div>
  );
}
