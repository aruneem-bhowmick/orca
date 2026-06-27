import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import apiClient from "@/api/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { formatDate, formatElapsed } from "@/lib/utils";
import { useWebSocket } from "@/hooks/useWebSocket";
import type { Experiment, Bookmark, MetricUpdate } from "@/api/types";

/**
 * Row component for the live metric table.
 *
 * Displays a single metric update as a table row with epoch number,
 * loss, accuracy (if available), learning rate (if available), and
 * timestamp.
 *
 * @param props.update - The metric update to display.
 */
function MetricRow({ update }: { update: MetricUpdate }) {
  return (
    <tr className="border-b last:border-0 text-sm">
      <td className="px-4 py-2 tabular-nums">{update.epoch}</td>
      <td className="px-4 py-2 tabular-nums">{update.loss.toFixed(4)}</td>
      <td className="px-4 py-2 tabular-nums">
        {update.accuracy !== null ? (update.accuracy * 100).toFixed(2) + "%" : "—"}
      </td>
      <td className="px-4 py-2 tabular-nums">
        {update.learning_rate !== null ? update.learning_rate.toExponential(2) : "—"}
      </td>
      <td className="px-4 py-2 text-muted-foreground">{formatDate(update.timestamp)}</td>
    </tr>
  );
}

/**
 * Live metrics section shown when an experiment is actively running.
 *
 * Opens a WebSocket connection via `useWebSocket` and renders:
 * - A line chart with loss and accuracy curves that updates as new
 *   metric frames arrive.
 * - Control buttons (Pause, Resume, Cancel) that send JSON control
 *   messages over the same socket.
 * - A scrollable table of all received metric updates in reverse
 *   chronological order.
 *
 * @param props.experimentId - The running experiment whose metrics to stream.
 */
function LiveMetricsSection({ experimentId }: { experimentId: string }) {
  const { messages, isConnected, send, close } = useWebSocket(experimentId);

  const chartData = messages.map((m) => ({
    epoch: m.epoch,
    loss: m.loss,
    accuracy: m.accuracy,
  }));

  return (
    <div className="space-y-6" data-testid="live-metrics">
      <div className="flex items-center gap-3">
        <span
          className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ${
            isConnected
              ? "bg-green-100 text-green-800"
              : "bg-muted text-muted-foreground"
          }`}
          data-testid="ws-status"
        >
          <span
            className={`h-1.5 w-1.5 rounded-full ${isConnected ? "animate-pulse bg-green-500" : "bg-muted-foreground"}`}
            aria-hidden="true"
          />
          {isConnected ? "Live" : "Reconnecting…"}
        </span>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => send({ action: "pause" })}
            disabled={!isConnected}
            data-testid="pause-btn"
          >
            Pause
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => send({ action: "resume" })}
            disabled={!isConnected}
            data-testid="resume-btn"
          >
            Resume
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              send({ action: "cancel" });
              close();
            }}
            data-testid="cancel-btn"
          >
            Cancel
          </Button>
        </div>
      </div>

      <div data-testid="metric-chart">
        {messages.length === 0 ? (
          <p className="text-sm text-muted-foreground" data-testid="no-metrics-yet">
            Waiting for metric data…
          </p>
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" label={{ value: "Epoch", position: "insideBottom", offset: -4 }} />
              <YAxis yAxisId="loss" orientation="left" />
              <YAxis yAxisId="accuracy" orientation="right" domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Line
                yAxisId="loss"
                type="monotone"
                dataKey="loss"
                stroke="#ef4444"
                dot={false}
                name="Loss"
              />
              <Line
                yAxisId="accuracy"
                type="monotone"
                dataKey="accuracy"
                stroke="#22c55e"
                dot={false}
                name="Accuracy"
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {messages.length > 0 && (
        <div className="overflow-auto rounded-lg border" data-testid="metric-table">
          <table className="w-full text-sm">
            <thead className="border-b bg-muted/50 text-left">
              <tr>
                <th className="px-4 py-2 font-medium">Epoch</th>
                <th className="px-4 py-2 font-medium">Loss</th>
                <th className="px-4 py-2 font-medium">Accuracy</th>
                <th className="px-4 py-2 font-medium">LR</th>
                <th className="px-4 py-2 font-medium">Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {[...messages].reverse().map((m) => (
                <MetricRow key={m.epoch} update={m} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/**
 * Final-metrics section shown when an experiment has completed.
 *
 * Renders the stored `metrics` object from the experiment record as a
 * summary grid, and a training-curve chart if individual epoch metrics
 * are available. Also displays a placeholder download link for the model
 * artifact.
 *
 * @param props.experiment - The completed experiment record.
 */
function CompletedMetricsSection({ experiment }: { experiment: Experiment }) {
  const metrics = experiment.metrics ?? {};

  return (
    <div className="space-y-6" data-testid="completed-metrics">
      {Object.keys(metrics).length > 0 ? (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3" data-testid="final-metrics-grid">
          {Object.entries(metrics).map(([key, value]) => (
            <div key={key} className="rounded-lg border p-4">
              <p className="text-xs text-muted-foreground">{key}</p>
              <p className="mt-1 text-2xl font-bold tabular-nums">
                {typeof value === "number" ? value.toFixed(4) : String(value)}
              </p>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">No final metrics recorded.</p>
      )}

      {experiment.mlflow_run_id && (
        <div data-testid="artifact-section">
          <p className="text-sm font-medium">Model Artifact</p>
          <p className="mt-1 text-xs text-muted-foreground">
            MLflow run ID:{" "}
            <code className="font-mono">{experiment.mlflow_run_id}</code>
          </p>
          <a
            href={`/api/v1/orcalab/experiments/${experiment.experiment_id}/artifact`}
            className="mt-2 inline-block text-sm text-primary underline-offset-4 hover:underline"
            data-testid="artifact-download"
          >
            Download model artifact
          </a>
        </div>
      )}
    </div>
  );
}

/**
 * OrcaLab experiment detail page.
 *
 * Fetches the experiment record from `GET /orcalab/experiments/:id`.
 * When the experiment is `running`, a `LiveMetricsSection` opens a
 * WebSocket connection and streams real-time loss and accuracy charts.
 * When `completed`, a `CompletedMetricsSection` shows the final stored
 * metrics and an artifact download link.
 *
 * A bookmark toggle at the top calls `POST /bookmarks` or
 * `DELETE /bookmarks/:id` to persist the experiment in the user's
 * bookmarks.
 */
export function ExperimentDetail() {
  const { id: experimentId } = useParams<{ id: string }>();
  const [bookmarked, setBookmarked] = useState(false);
  const [bookmarkId, setBookmarkId] = useState<string | null>(null);
  const isBookmarkPending = useRef(false);

  useEffect(() => {
    async function checkBookmarkStatus() {
      if (!experimentId) return;
      try {
        const res = await apiClient.get<any>("/bookmarks?per_page=100");
        const items = res.data?.items;
        if (Array.isArray(items)) {
          const existing = items.find(
            (b: any) =>
              b.resource_type === "experiment" &&
              b.resource_id === experimentId,
          );
          if (existing) {
            setBookmarked(true);
            setBookmarkId(existing.id);
          } else {
            setBookmarked(false);
            setBookmarkId(null);
          }
        }
      } catch {
        // ignore
      }
    }
    checkBookmarkStatus();
  }, [experimentId]);

  const {
    data: experiment,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["orcalab-experiment", experimentId],
    queryFn: async () => {
      const res = await apiClient.get<Experiment>(
        `/orcalab/experiments/${experimentId}`,
      );
      return res.data;
    },
    enabled: !!experimentId,
    staleTime: 60_000,
    refetchInterval: (query) =>
      query.state.data?.status === "running" ? 15_000 : false,
  });

  async function handleBookmarkToggle() {
    if (isBookmarkPending.current) return;
    isBookmarkPending.current = true;
    try {
      if (bookmarked && bookmarkId) {
        await apiClient.delete(`/bookmarks/${bookmarkId}`);
        setBookmarked(false);
        setBookmarkId(null);
      } else {
        const res = await apiClient.post<Bookmark>("/bookmarks", {
          resource_type: "experiment",
          resource_id: experimentId,
        });
        setBookmarked(true);
        setBookmarkId(res.data.id);
      }
    } catch {
      /* ignore transient bookmark errors */
    } finally {
      isBookmarkPending.current = false;
    }
  }

  return (
    <div className="space-y-8">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold">Experiment Detail</h1>
          <p className="mt-1 text-muted-foreground">
            OrcaLab training run metadata and live metrics.
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

      {/* Metadata card */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Metadata</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <p className="text-muted-foreground" data-testid="exp-loading">
              Loading experiment…
            </p>
          ) : isError || !experiment ? (
            <p className="text-destructive" data-testid="exp-error">
              Failed to load experiment.
            </p>
          ) : (
            <dl
              className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-3"
              data-testid="exp-metadata"
            >
              <div>
                <dt className="text-muted-foreground">Name</dt>
                <dd className="font-medium">{experiment.name}</dd>
              </div>
              <div>
                <dt className="text-muted-foreground">Status</dt>
                <dd>{experiment.status}</dd>
              </div>
              <div>
                <dt className="text-muted-foreground">Task ID</dt>
                <dd className="font-mono text-xs">{experiment.task_id}</dd>
              </div>
              <div>
                <dt className="text-muted-foreground">Model ID</dt>
                <dd className="font-mono text-xs">{experiment.model_id}</dd>
              </div>
              <div>
                <dt className="text-muted-foreground">Started</dt>
                <dd>
                  {experiment.started_at
                    ? formatDate(experiment.started_at)
                    : "—"}
                </dd>
              </div>
              <div>
                <dt className="text-muted-foreground">Elapsed</dt>
                <dd>
                  {experiment.started_at
                    ? formatElapsed(
                        experiment.started_at,
                        experiment.completed_at,
                      )
                    : "—"}
                </dd>
              </div>
              <div>
                <dt className="text-muted-foreground">Created</dt>
                <dd>{formatDate(experiment.created_at)}</dd>
              </div>
            </dl>
          )}
        </CardContent>
      </Card>

      {/* Live or completed metrics */}
      {experiment && (
        <div>
          <h2 className="mb-4 text-lg font-semibold">
            {experiment.status === "running" ? "Live Metrics" : "Final Metrics"}
          </h2>
          {experiment.status === "running" && experimentId ? (
            <LiveMetricsSection experimentId={experimentId} />
          ) : experiment.status === "completed" ? (
            <CompletedMetricsSection experiment={experiment} />
          ) : (
            <p className="text-sm text-muted-foreground" data-testid="no-metrics-msg">
              {experiment.status === "failed"
                ? "Experiment failed. No metrics available."
                : "Experiment has not started yet."}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
