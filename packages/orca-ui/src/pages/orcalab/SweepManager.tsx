import { useState, useEffect, Fragment } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
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
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { formatDate } from "@/lib/utils";
import type { Sweep, Task, CreateSweepRequest, ExperimentStatus } from "@/api/types";

/** Supported hyperparameter search strategies. */
const SEARCH_STRATEGIES = ["random", "grid", "bayesian"] as const;
type SearchStrategy = (typeof SEARCH_STRATEGIES)[number];

/** Status badge colour classes keyed by lifecycle status. */
const STATUS_COLORS: Record<ExperimentStatus, string> = {
  pending: "bg-muted text-muted-foreground",
  running: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  completed: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
  failed: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
};

/**
 * Small pill badge indicating a sweep's current status.
 *
 * @param props.status - Lifecycle status to display.
 */
function StatusBadge({ status }: { status: ExperimentStatus }) {
  return (
    <span
      className={`inline-flex rounded-full px-2.5 py-0.5 text-xs font-medium ${STATUS_COLORS[status]}`}
    >
      {status}
    </span>
  );
}

/**
 * Progress bar showing how many trials have completed out of the total.
 *
 * @param props.completed - Number of finished trials.
 * @param props.total - Total number of planned trials.
 */
function TrialProgress({
  completed,
  total,
}: {
  completed: number;
  total: number;
}) {
  const pct = total > 0 ? Math.round((completed / total) * 100) : 0;
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>{completed} / {total} trials</span>
        <span>{pct}%</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-primary transition-all"
          style={{ width: `${pct}%` }}
          data-testid="trial-progress-bar"
        />
      </div>
    </div>
  );
}

/**
 * Parallel-coordinates-style chart showing metric values per trial.
 *
 * Since Recharts does not natively support parallel coordinates, this
 * renders a simple line chart where each trial is an X-axis tick and
 * each tracked metric is a separate `Line`. The best trial is
 * highlighted by a reference marker colour in the legend.
 *
 * Renders nothing when `results` is empty or `null`.
 *
 * @param props.results - Array of completed sweep trials.
 * @param props.bestTrial - 1-based index of the best trial (highlighted).
 */
function SweepResultsChart({
  results,
  bestTrial,
}: {
  results: NonNullable<Sweep["results"]>;
  bestTrial: number | null;
}) {
  if (results.length === 0) return null;

  // Collect unique metric names across all trials
  const metricKeys = Array.from(
    new Set(results.flatMap((r) => Object.keys(r.metrics))),
  );

  const chartData = results.map((trial) => ({
    trial: trial.trial_id,
    ...trial.metrics,
  }));

  const COLORS = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#06b6d4"];

  return (
    <div data-testid="sweep-chart">
      <p className="mb-2 text-sm text-muted-foreground">
        Metric values per trial
        {bestTrial !== null && ` (best: trial ${bestTrial})`}
      </p>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="trial" label={{ value: "Trial", position: "insideBottom", offset: -4 }} />
          <YAxis />
          <Tooltip />
          <Legend />
          {metricKeys.map((key, i) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={COLORS[i % COLORS.length]}
              dot={(props) => {
                const isBest = props.payload?.trial === bestTrial;
                return (
                  <circle
                    key={`dot-${props.index}`}
                    cx={props.cx}
                    cy={props.cy}
                    r={isBest ? 6 : 3}
                    fill={isBest ? "#f59e0b" : COLORS[i % COLORS.length]}
                    stroke={isBest ? "#b45309" : "none"}
                    strokeWidth={isBest ? 2 : 0}
                  />
                );
              }}
              name={key}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/**
 * Expanded detail card shown below a sweep row when it is selected.
 *
 * Displays trial progress, the results chart, and a summary table of
 * each trial's hyperparameter values and metric scores.
 *
 * @param props.sweep - The selected sweep to detail.
 */
function SweepDetailPanel({ sweep }: { sweep: Sweep }) {
  const results = sweep.results ?? [];
  const paramKeys = results[0] ? Object.keys(results[0].params) : [];
  const metricKeys = results[0] ? Object.keys(results[0].metrics) : [];

  return (
    <div className="space-y-4 px-4 pb-4" data-testid={`sweep-detail-${sweep.sweep_id}`}>
      <TrialProgress
        completed={sweep.completed_trials}
        total={sweep.n_trials}
      />

      {results.length > 0 ? (
        <>
          <SweepResultsChart results={results} bestTrial={sweep.best_trial} />

          <div className="overflow-auto rounded-lg border" data-testid="sweep-trials-table">
            <table className="w-full text-sm">
              <thead className="border-b bg-muted/50 text-left">
                <tr>
                  <th className="px-3 py-2 font-medium">Trial</th>
                  {paramKeys.map((k) => (
                    <th key={k} className="px-3 py-2 font-medium">{k}</th>
                  ))}
                  {metricKeys.map((k) => (
                    <th key={k} className="px-3 py-2 font-medium text-primary">{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.map((trial) => (
                  <tr
                    key={trial.trial_id}
                    className={`border-b last:border-0 ${
                      trial.trial_id === sweep.best_trial
                        ? "bg-amber-50 dark:bg-amber-950"
                        : ""
                    }`}
                    data-testid={`trial-row-${trial.trial_id}`}
                  >
                    <td className="px-3 py-2 font-medium">
                      {trial.trial_id}
                      {trial.trial_id === sweep.best_trial && (
                        <span className="ml-1 text-xs text-amber-600" aria-label="best trial">
                          ★
                        </span>
                      )}
                    </td>
                    {paramKeys.map((key) => {
                      const v = trial.params[key];
                      return (
                        <td key={key} className="px-3 py-2 tabular-nums">
                          {v !== undefined ? String(v) : "—"}
                        </td>
                      );
                    })}
                    {metricKeys.map((key) => {
                      const v = trial.metrics[key];
                      return (
                        <td key={key} className="px-3 py-2 tabular-nums font-medium text-primary">
                          {v !== undefined ? (typeof v === "number" ? v.toFixed(4) : String(v)) : "—"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <p className="text-sm text-muted-foreground" data-testid="no-trials-yet">
          No completed trials yet.
        </p>
      )}
    </div>
  );
}

/**
 * Modal form for launching a new hyperparameter sweep.
 *
 * The user selects an existing task from a dropdown, picks a search
 * strategy, specifies the number of trials, and optionally enables
 * OrcaMind-prior seeding for the initial trial configurations.
 *
 * @param props.tasks - List of available OrcaMind tasks to select from.
 * @param props.onClose - Callback invoked when the form is dismissed.
 * @param props.onSubmit - Callback invoked with the validated request data.
 * @param props.isSubmitting - Whether the creation request is in flight.
 */
function NewSweepDialog({
  tasks,
  onClose,
  onSubmit,
  isSubmitting,
}: {
  tasks: Task[];
  onClose: () => void;
  onSubmit: (data: CreateSweepRequest) => void;
  isSubmitting: boolean;
}) {
  const [taskId, setTaskId] = useState(tasks[0]?.task_id ?? "");
  const [strategy, setStrategy] = useState<SearchStrategy>("random");
  const [nTrials, setNTrials] = useState(20);
  const [useOrcaMindPriors, setUseOrcaMindPriors] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!taskId && tasks.length > 0) {
      setTaskId(tasks[0].task_id);
    }
  }, [tasks, taskId]);

  function validate(): boolean {
    const e: Record<string, string> = {};
    if (!taskId) e.taskId = "Task is required";
    if (!nTrials || nTrials < 1) {
      e.nTrials = "Number of trials must be at least 1";
    } else if (!Number.isInteger(nTrials)) {
      e.nTrials = "Number of trials must be a whole number";
    }
    setErrors(e);
    return Object.keys(e).length === 0;
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!validate()) return;
    onSubmit({
      task_id: taskId,
      search_strategy: strategy,
      n_trials: nTrials,
      use_orcamind_priors: useOrcaMindPriors,
    });
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      data-testid="new-sweep-dialog"
    >
      <div className="w-full max-w-lg rounded-lg bg-background p-6 shadow-lg">
        <h2 className="mb-4 text-xl font-semibold">New Sweep</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label
              htmlFor="sweep-task"
              className="mb-1 block text-sm font-medium"
            >
              Task
            </label>
            <select
              id="sweep-task"
              value={taskId}
              onChange={(e) => setTaskId(e.target.value)}
              className="w-full rounded-md border border-input bg-background px-3 py-1.5 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              data-testid="sweep-task-select"
            >
              {tasks.map((t) => (
                <option key={t.task_id} value={t.task_id}>
                  {t.name} ({t.domain})
                </option>
              ))}
            </select>
            {errors.taskId && (
              <p className="mt-1 text-xs text-destructive">{errors.taskId}</p>
            )}
          </div>

          <div>
            <label
              htmlFor="sweep-strategy"
              className="mb-1 block text-sm font-medium"
            >
              Search Strategy
            </label>
            <select
              id="sweep-strategy"
              value={strategy}
              onChange={(e) => setStrategy(e.target.value as SearchStrategy)}
              className="w-full rounded-md border border-input bg-background px-3 py-1.5 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              data-testid="sweep-strategy-select"
            >
              {SEARCH_STRATEGIES.map((s) => (
                <option key={s} value={s}>
                  {s.charAt(0).toUpperCase() + s.slice(1)}
                </option>
              ))}
            </select>
          </div>

          <Input
            label="Number of Trials"
            type="number"
            value={nTrials}
            min={1}
            step="any"
            onChange={(e) => setNTrials(Number(e.target.value))}
            error={errors.nTrials}
            data-testid="sweep-n-trials"
          />

          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={useOrcaMindPriors}
              onChange={(e) => setUseOrcaMindPriors(e.target.checked)}
              className="rounded border border-input"
              data-testid="sweep-use-priors"
            />
            Use OrcaMind priors for initial trial configurations
          </label>

          <div className="flex justify-end gap-3 pt-2">
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={isSubmitting} data-testid="sweep-submit-btn">
              {isSubmitting ? "Launching…" : "Launch Sweep"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}

/**
 * OrcaLab sweep manager page.
 *
 * Fetches all sweeps from `GET /orcalab/sweeps` and lists them in a
 * table showing task, strategy, trial progress, and status. Clicking a
 * row expands an inline detail panel with a per-trial chart and results
 * table; clicking again collapses it.
 *
 * The "New Sweep" button opens a creation dialog. Available tasks are
 * loaded from `GET /orcamind/tasks` to populate the task dropdown.
 */
export function SweepManager() {
  const queryClient = useQueryClient();
  const [selectedSweepId, setSelectedSweepId] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);

  const { data: sweeps = [], isLoading, isError } = useQuery({
    queryKey: ["orcalab-sweeps"],
    queryFn: async () => {
      const res = await apiClient.get<Sweep[]>("/orcalab/sweeps");
      return res.data;
    },
    staleTime: 30_000,
  });

  const { data: tasks = [] } = useQuery({
    queryKey: ["orcamind-tasks"],
    queryFn: async () => {
      const res = await apiClient.get<Task[]>("/orcamind/tasks");
      return res.data;
    },
    staleTime: 60_000,
  });

  const { mutate: createSweep, isPending: isCreating } = useMutation({
    mutationFn: async (data: CreateSweepRequest) => {
      const res = await apiClient.post<Sweep>("/orcalab/sweeps", data);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["orcalab-sweeps"] });
      setDialogOpen(false);
    },
  });

  function toggleRow(sweepId: string) {
    setSelectedSweepId((prev) => (prev === sweepId ? null : sweepId));
  }

  return (
    <div>
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Sweeps</h1>
        <Button onClick={() => setDialogOpen(true)} data-testid="new-sweep-btn">
          New Sweep
        </Button>
      </div>
      <p className="mt-1 text-muted-foreground">
        Hyperparameter sweeps and their trial results.
      </p>

      <div className="mt-6 overflow-auto rounded-lg border">
        {isLoading ? (
          <p className="p-4 text-muted-foreground" data-testid="sweep-list-loading">
            Loading sweeps…
          </p>
        ) : isError ? (
          <p className="p-4 text-destructive" data-testid="sweep-list-error">
            Failed to load sweeps.
          </p>
        ) : sweeps.length === 0 ? (
          <p className="p-4 text-muted-foreground" data-testid="sweep-list-empty">
            No sweeps found.
          </p>
        ) : (
          <table className="w-full text-sm">
            <thead className="border-b bg-muted/50 text-left">
              <tr>
                <th className="px-4 py-3 font-medium">ID</th>
                <th className="px-4 py-3 font-medium">Task</th>
                <th className="px-4 py-3 font-medium">Strategy</th>
                <th className="px-4 py-3 font-medium">Trials</th>
                <th className="px-4 py-3 font-medium">Status</th>
                <th className="px-4 py-3 font-medium">Created</th>
              </tr>
            </thead>
            <tbody>
              {sweeps.map((sweep) => (
                <Fragment key={sweep.sweep_id}>
                  <tr
                    className="cursor-pointer border-b hover:bg-muted/50"
                    tabIndex={0}
                    onClick={() => toggleRow(sweep.sweep_id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        toggleRow(sweep.sweep_id);
                      }
                    }}
                    data-testid={`sweep-row-${sweep.sweep_id}`}
                  >
                    <td className="px-4 py-3 font-mono text-xs">
                      {sweep.sweep_id}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">
                      {sweep.task_id}
                    </td>
                    <td className="px-4 py-3">{sweep.search_strategy}</td>
                    <td className="px-4 py-3">
                      {sweep.completed_trials} / {sweep.n_trials}
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge status={sweep.status} />
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">
                      {formatDate(sweep.created_at)}
                    </td>
                  </tr>
                  {selectedSweepId === sweep.sweep_id && (
                    <tr key={`detail-${sweep.sweep_id}`} className="border-b">
                      <td colSpan={6} className="p-0">
                        <Card className="m-0 rounded-none border-0 border-t">
                           <CardHeader className="px-4 py-3">
                            <CardTitle className="text-sm font-medium">
                              Sweep Details
                            </CardTitle>
                          </CardHeader>
                          <CardContent className="p-0">
                            <SweepDetailPanel sweep={sweep} />
                          </CardContent>
                        </Card>
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {dialogOpen && (
        <NewSweepDialog
          tasks={tasks}
          onClose={() => setDialogOpen(false)}
          onSubmit={createSweep}
          isSubmitting={isCreating}
        />
      )}
    </div>
  );
}
