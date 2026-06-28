import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import {
  RadialBarChart,
  RadialBar,
  PolarAngleAxis,
  ResponsiveContainer,
} from "recharts";
import apiClient from "@/api/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { ROUTES } from "@/lib/constants";
import type {
  Task,
  TransferScoreRequest,
  TransferScoreResult,
  TransferRecommendRequest,
  TransferRecommendation,
  ExplainRequest,
  ExplainResult,
} from "@/api/types";

// ---------------------------------------------------------------------------
// Score gauge
// ---------------------------------------------------------------------------

/**
 * Circular gauge that visualises a transfer score as a filled radial arc.
 *
 * The score is expected to be in the [0, 1] range and is rendered as a
 * percentage-filled arc via Recharts `RadialBarChart`. A numeric label is
 * centred inside the chart.
 *
 * @param props.score - Transfer score value in [0, 1].
 */
function ScoreGauge({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const data = [{ value: pct, fill: pct >= 70 ? "#22c55e" : pct >= 40 ? "#f59e0b" : "#ef4444" }];
  return (
    <div className="flex flex-col items-center gap-1" data-testid="score-gauge">
      <div style={{ width: 140, height: 140 }}>
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            innerRadius="70%"
            outerRadius="100%"
            startAngle={90}
            endAngle={-270}
            data={data}
            barSize={14}
          >
            <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
            <RadialBar dataKey="value" cornerRadius={6} background={{ fill: "hsl(var(--muted))" }} />
          </RadialBarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-3xl font-bold tabular-nums" data-testid="score-value">
        {pct}%
      </p>
      <p className="text-sm text-muted-foreground">Transfer Score</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Recommendation card
// ---------------------------------------------------------------------------

/**
 * A single ranked transfer recommendation card.
 *
 * Displays the source task name, transfer score, strategy label, and an
 * "Explain" button that lazily fetches the LLM-generated explanation.
 * An "Apply Transfer" button navigates to the OrcaLab experiment list
 * pre-filled with the target task ID.
 *
 * @param props.rec - The transfer recommendation to display.
 * @param props.targetTaskId - Target task ID used for explain and apply actions.
 * @param props.onApply - Callback invoked when the user clicks "Apply Transfer".
 */
function RecommendationCard({
  rec,
  targetTaskId,
  onApply,
}: {
  rec: TransferRecommendation;
  targetTaskId: string;
  onApply: (rec: TransferRecommendation) => void;
}) {
  const [explanation, setExplanation] = useState<string | null>(null);
  const [explainError, setExplainError] = useState(false);

  const { mutate: explain, isPending: isExplaining } = useMutation({
    mutationFn: async () => {
      const payload: ExplainRequest = {
        source_task_id: rec.source_task_id,
        target_task_id: targetTaskId,
        strategy: rec.strategy,
      };
      const res = await apiClient.post<ExplainResult>("/orcanet/explain", payload);
      return res.data;
    },
    onSuccess: (data) => {
      setExplanation(data.explanation);
      setExplainError(false);
    },
    onError: () => {
      setExplainError(true);
    },
  });

  return (
    <Card data-testid={`rec-card-${rec.source_task_id}`}>
      <CardHeader>
        <CardTitle className="text-base">{rec.source_task_name}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center gap-4 text-sm">
          <span>
            <span className="text-muted-foreground">Score: </span>
            <span className="font-medium" data-testid="rec-score">
              {(rec.transfer_score * 100).toFixed(0)}%
            </span>
          </span>
          <span>
            <span className="text-muted-foreground">Strategy: </span>
            <span
              className="rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary"
              data-testid="rec-strategy"
            >
              {rec.strategy}
            </span>
          </span>
        </div>

        {explanation && (
          <div
            className="rounded-md bg-muted p-3 text-sm leading-relaxed"
            data-testid="explanation-panel"
          >
            {explanation}
          </div>
        )}
        {explainError && (
          <p className="text-sm text-destructive" data-testid="explain-error">
            Failed to load explanation.
          </p>
        )}

        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => explain()}
            disabled={isExplaining}
            data-testid="explain-btn"
          >
            {isExplaining ? "Loading…" : "Explain"}
          </Button>
          <Button
            size="sm"
            onClick={() => onApply(rec)}
            data-testid="apply-btn"
          >
            Apply Transfer
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// TransferExplorer page
// ---------------------------------------------------------------------------

/**
 * OrcaNet Transfer Explorer page.
 *
 * Allows the user to:
 * 1. Select a source task and a target task from their OrcaMind task list.
 * 2. Score the transfer between them via `POST /orcanet/transfer/score`,
 *    displayed as a radial gauge.
 * 3. Request ranked transfer recommendations via
 *    `POST /orcanet/transfer/recommend`, rendered as expandable cards.
 * 4. Fetch an LLM-generated explanation for any recommendation via
 *    `POST /orcanet/explain`.
 * 5. Apply a recommendation, which navigates to the OrcaLab experiment list
 *    pre-filled with the target task ID and source model config.
 */
export function TransferExplorer() {
  const navigate = useNavigate();

  const [sourceTaskId, setSourceTaskId] = useState("");
  const [targetTaskId, setTargetTaskId] = useState("");
  const [scoreResult, setScoreResult] = useState<TransferScoreResult | null>(null);
  const [recommendations, setRecommendations] = useState<TransferRecommendation[]>([]);
  const [scoreError, setScoreError] = useState(false);
  const [recommendError, setRecommendError] = useState(false);

  const { data: tasks = [], isLoading: tasksLoading } = useQuery({
    queryKey: ["orcamind-tasks"],
    queryFn: async () => {
      const res = await apiClient.get<Task[]>("/orcamind/tasks");
      return res.data;
    },
    staleTime: 30_000,
  });

  const { mutate: scoreTransfer, isPending: isScoring } = useMutation({
    mutationFn: async () => {
      const payload: TransferScoreRequest = { source_task_id: sourceTaskId, target_task_id: targetTaskId };
      const res = await apiClient.post<TransferScoreResult>("/orcanet/transfer/score", payload);
      return res.data;
    },
    onSuccess: (data) => {
      setScoreResult(data);
      setScoreError(false);
    },
    onError: () => {
      setScoreError(true);
    },
  });

  const { mutate: getRecommendations, isPending: isRecommending } = useMutation({
    mutationFn: async () => {
      const payload: TransferRecommendRequest = { target_task_id: targetTaskId };
      const res = await apiClient.post<TransferRecommendation[]>("/orcanet/transfer/recommend", payload);
      return res.data;
    },
    onSuccess: (data) => {
      setRecommendations(data);
      setRecommendError(false);
    },
    onError: () => {
      setRecommendError(true);
    },
  });

  /** Navigate to the OrcaLab experiment list pre-filled with the target task. */
  function handleApply(rec: TransferRecommendation) {
    const modelId = rec.config?.model_id as string | undefined;
    const params = new URLSearchParams({ task_id: targetTaskId });
    if (modelId) params.set("model_id", modelId);
    navigate(`${ROUTES.ORCALAB_EXPERIMENTS}?${params.toString()}`);
  }

  const canScore = sourceTaskId && targetTaskId && sourceTaskId !== targetTaskId;
  const canRecommend = !!targetTaskId;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Transfer Explorer</h1>
        <p className="mt-1 text-muted-foreground">
          Score and explore knowledge transfer between OrcaMind tasks.
        </p>
      </div>

      {/* Task selection */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Select Tasks</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-1">
            <label htmlFor="source-task" className="text-sm font-medium">
              Source Task
            </label>
            <select
              id="source-task"
              value={sourceTaskId}
              onChange={(e) => setSourceTaskId(e.target.value)}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              data-testid="source-task-select"
              disabled={tasksLoading}
            >
              <option value="">— Select source task —</option>
              {tasks.map((t) => (
                <option key={t.task_id} value={t.task_id}>
                  {t.name} ({t.domain})
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-1">
            <label htmlFor="target-task" className="text-sm font-medium">
              Target Task
            </label>
            <select
              id="target-task"
              value={targetTaskId}
              onChange={(e) => setTargetTaskId(e.target.value)}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              data-testid="target-task-select"
              disabled={tasksLoading}
            >
              <option value="">— Select target task —</option>
              {tasks.map((t) => (
                <option key={t.task_id} value={t.task_id}>
                  {t.name} ({t.domain})
                </option>
              ))}
            </select>
          </div>
        </CardContent>
      </Card>

      {/* Actions */}
      <div className="flex flex-wrap gap-3">
        <Button
          onClick={() => scoreTransfer()}
          disabled={!canScore || isScoring}
          data-testid="score-btn"
        >
          {isScoring ? "Scoring…" : "Score Transfer"}
        </Button>
        <Button
          variant="outline"
          onClick={() => getRecommendations()}
          disabled={!canRecommend || isRecommending}
          data-testid="recommend-btn"
        >
          {isRecommending ? "Loading…" : "Get Recommendations"}
        </Button>
      </div>

      {/* Score result */}
      {scoreError && (
        <p className="text-sm text-destructive" data-testid="score-error">
          Failed to score transfer. Please try again.
        </p>
      )}
      {scoreResult && (
        <Card data-testid="score-result-card">
          <CardContent className="pt-6">
            <ScoreGauge score={scoreResult.score} />
          </CardContent>
        </Card>
      )}

      {/* Recommendations */}
      {recommendError && (
        <p className="text-sm text-destructive" data-testid="recommend-error">
          Failed to load recommendations.
        </p>
      )}
      {recommendations.length > 0 && (
        <div className="space-y-3" data-testid="recommendations-list">
          <h2 className="text-lg font-semibold">Recommendations</h2>
          {recommendations.map((rec) => (
            <RecommendationCard
              key={rec.source_task_id}
              rec={rec}
              targetTaskId={targetTaskId}
              onApply={handleApply}
            />
          ))}
        </div>
      )}
    </div>
  );
}
