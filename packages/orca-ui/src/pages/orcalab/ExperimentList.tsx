import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useSearchParams } from "react-router-dom";
import apiClient from "@/api/client";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { formatDate, formatElapsed } from "@/lib/utils";
import { ROUTES } from "@/lib/constants";
import type {
  Experiment,
  ExperimentStatus,
  CreateExperimentRequest,
} from "@/api/types";

/** Status badge colour classes keyed by experiment status. */
const STATUS_COLORS: Record<ExperimentStatus, string> = {
  pending: "bg-muted text-muted-foreground",
  running: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  completed: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
  failed: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
};

/** All selectable status filter values including the "all" sentinel. */
const STATUS_FILTERS = ["all", "pending", "running", "completed", "failed"] as const;
type StatusFilter = (typeof STATUS_FILTERS)[number];

/**
 * Colour-coded pill badge for an experiment status.
 *
 * @param props.status - The lifecycle status to display.
 */
function StatusBadge({ status }: { status: ExperimentStatus }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ${STATUS_COLORS[status]}`}
      data-testid={`status-badge-${status}`}
    >
      {status === "running" && (
        <span
          className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue-500"
          aria-hidden="true"
        />
      )}
      {status}
    </span>
  );
}

/**
 * Modal form for creating a new OrcaLab experiment.
 *
 * Requires a task ID and a model ID; training config is optional and
 * defaults to an empty object when omitted. The `task_id` can be
 * pre-filled via the `?task_id` query param from the OrcaMind
 * recommendation flow.
 *
 * @param props.defaultTaskId - Optional pre-filled task ID.
 * @param props.onClose - Callback invoked when the dialog is dismissed.
 * @param props.onSubmit - Callback invoked with the validated form data.
 * @param props.isSubmitting - Whether the create request is in flight.
 */
function NewExperimentDialog({
  defaultTaskId,
  onClose,
  onSubmit,
  isSubmitting,
}: {
  defaultTaskId?: string;
  onClose: () => void;
  onSubmit: (data: CreateExperimentRequest) => void;
  isSubmitting: boolean;
}) {
  const [form, setForm] = useState<CreateExperimentRequest>({
    task_id: defaultTaskId ?? "",
    model_id: "",
  });
  const [errors, setErrors] = useState<Partial<Record<keyof CreateExperimentRequest, string>>>({});

  function validate(): boolean {
    const e: Partial<Record<keyof CreateExperimentRequest, string>> = {};
    if (!form.task_id.trim()) e.task_id = "Task ID is required";
    if (!form.model_id.trim()) e.model_id = "Model ID is required";
    setErrors(e);
    return Object.keys(e).length === 0;
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (validate()) onSubmit(form);
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      data-testid="new-experiment-dialog"
    >
      <div className="w-full max-w-lg rounded-lg bg-background p-6 shadow-lg">
        <h2 className="mb-4 text-xl font-semibold">New Experiment</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <Input
            label="Task ID"
            value={form.task_id}
            onChange={(e) => setForm({ ...form, task_id: e.target.value })}
            error={errors.task_id}
            data-testid="exp-task-id"
          />
          <Input
            label="Model ID"
            value={form.model_id}
            onChange={(e) => setForm({ ...form, model_id: e.target.value })}
            error={errors.model_id}
            data-testid="exp-model-id"
          />
          <div className="flex justify-end gap-3 pt-2">
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={isSubmitting} data-testid="exp-submit-btn">
              {isSubmitting ? "Creating…" : "Create Experiment"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}

/**
 * OrcaLab experiment list page.
 *
 * Fetches all experiments from `GET /orcalab/experiments` and displays
 * them in a table. A status-filter dropdown narrows the view to
 * `pending`, `running`, `completed`, or `failed` experiments. Running
 * experiments show a pulsing indicator in their status badge.
 *
 * Clicking a row navigates to the experiment detail page. The
 * "New Experiment" button opens a creation dialog; the `?task_id` and
 * `?model_id` query params pre-fill the form when arriving from the
 * OrcaMind task-detail or recommendations pages.
 */
export function ExperimentList() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [searchParams] = useSearchParams();

  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [dialogOpen, setDialogOpen] = useState(false);

  const prefillTaskId = searchParams.get("task_id") ?? undefined;

  const {
    data: experiments = [],
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["orcalab-experiments"],
    queryFn: async () => {
      const res = await apiClient.get<Experiment[]>("/orcalab/experiments");
      return res.data;
    },
    staleTime: 30_000,
  });

  const { mutate: createExperiment, isPending: isCreating } = useMutation({
    mutationFn: async (data: CreateExperimentRequest) => {
      const res = await apiClient.post<Experiment>("/orcalab/experiments", data);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["orcalab-experiments"] });
      setDialogOpen(false);
    },
  });

  const filtered =
    statusFilter === "all"
      ? experiments
      : experiments.filter((e) => e.status === statusFilter);

  return (
    <div>
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Experiments</h1>
        <Button onClick={() => setDialogOpen(true)} data-testid="new-experiment-btn">
          New Experiment
        </Button>
      </div>
      <p className="mt-1 text-muted-foreground">
        OrcaLab training runs and their live status.
      </p>

      <div className="mt-4 flex items-center gap-3">
        <label htmlFor="status-filter" className="text-sm font-medium">
          Status
        </label>
        <select
          id="status-filter"
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value as StatusFilter)}
          className="rounded-md border border-input bg-background px-3 py-1.5 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          data-testid="status-filter"
        >
          {STATUS_FILTERS.map((s) => (
            <option key={s} value={s}>
              {s === "all" ? "All statuses" : s.charAt(0).toUpperCase() + s.slice(1)}
            </option>
          ))}
        </select>
      </div>

      <div className="mt-4 overflow-auto rounded-lg border">
        {isLoading ? (
          <p className="p-4 text-muted-foreground" data-testid="exp-list-loading">
            Loading experiments…
          </p>
        ) : isError ? (
          <p className="p-4 text-destructive" data-testid="exp-list-error">
            Failed to load experiments.
          </p>
        ) : filtered.length === 0 ? (
          <p className="p-4 text-muted-foreground" data-testid="exp-list-empty">
            No experiments found.
          </p>
        ) : (
          <table className="w-full text-sm" data-testid="exp-table">
            <thead className="border-b bg-muted/50 text-left">
              <tr>
                <th className="px-4 py-3 font-medium">Name / ID</th>
                <th className="px-4 py-3 font-medium">Task</th>
                <th className="px-4 py-3 font-medium">Model</th>
                <th className="px-4 py-3 font-medium">Status</th>
                <th className="px-4 py-3 font-medium">Started</th>
                <th className="px-4 py-3 font-medium">Elapsed</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((exp) => (
                <tr
                  key={exp.experiment_id}
                  className="cursor-pointer border-b last:border-0 hover:bg-muted/50"
                  tabIndex={0}
                  onClick={() =>
                    navigate(
                      `${ROUTES.ORCALAB_EXPERIMENTS}/${exp.experiment_id}`,
                    )
                  }
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      navigate(
                        `${ROUTES.ORCALAB_EXPERIMENTS}/${exp.experiment_id}`,
                      );
                    }
                  }}
                  data-testid={`exp-row-${exp.experiment_id}`}
                >
                  <td className="px-4 py-3">
                    <p className="font-medium">{exp.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {exp.experiment_id}
                    </p>
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {exp.task_id}
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {exp.model_id}
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={exp.status} />
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {exp.started_at ? formatDate(exp.started_at) : "—"}
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {exp.started_at
                      ? formatElapsed(exp.started_at, exp.completed_at)
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {dialogOpen && (
        <NewExperimentDialog
          defaultTaskId={prefillTaskId}
          onClose={() => setDialogOpen(false)}
          onSubmit={createExperiment}
          isSubmitting={isCreating}
        />
      )}
    </div>
  );
}
