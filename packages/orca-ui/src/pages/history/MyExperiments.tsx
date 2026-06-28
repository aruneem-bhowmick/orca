import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import apiClient from "@/api/client";
import { formatDate } from "@/lib/utils";
import { ROUTES } from "@/lib/constants";
import type { ActivityLogEntry, ExperimentStatus, PaginatedResponse } from "@/api/types";

// ---------------------------------------------------------------------------
// Status badge
// ---------------------------------------------------------------------------

/** Colour classes keyed by experiment status string. */
const STATUS_COLORS: Record<string, string> = {
  pending: "bg-muted text-muted-foreground",
  running: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  completed: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
  failed: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
};

/**
 * Derives a display status from an activity log entry's action string.
 *
 * Maps known action suffixes ("_started", "_completed", "_failed") to
 * lifecycle status values. Returns `null` when no mapping is found.
 *
 * @param action - The activity log action string (e.g. "experiment_completed").
 * @returns A human-readable status label, or null.
 */
function statusFromAction(action: string): ExperimentStatus | null {
  if (action.includes("started") || action.includes("running")) return "running";
  if (action.includes("completed")) return "completed";
  if (action.includes("failed")) return "failed";
  if (action.includes("pending")) return "pending";
  return null;
}

/**
 * Status badge derived from an activity log entry's action string.
 *
 * @param props.action - The activity log action string.
 */
function StatusBadge({ action }: { action: string }) {
  const status = statusFromAction(action);
  if (!status) return null;
  const colorClass = STATUS_COLORS[status] ?? "bg-muted text-muted-foreground";
  return (
    <span
      className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${colorClass}`}
      data-testid={`exp-status-badge-${status}`}
    >
      {status}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Experiment entry row
// ---------------------------------------------------------------------------

/**
 * A single experiment-related activity entry row.
 *
 * Clicking the row navigates to the OrcaLab experiment detail page for the
 * referenced resource. Includes a status badge derived from the action string.
 *
 * @param props.entry - The activity log entry to render.
 * @param props.onNavigate - Callback invoked with the experiment ID.
 */
function ExperimentEntryRow({
  entry,
  onNavigate,
}: {
  entry: ActivityLogEntry;
  onNavigate: (experimentId: string) => void;
}) {
  const isClickable = !!entry.resource_id;

  const content = (
    <>
      <div className="space-y-1 text-left">
        <div className="flex flex-wrap items-center gap-2">
          <p className="font-medium" data-testid="entry-action">
            {entry.action}
          </p>
          <StatusBadge action={entry.action} />
        </div>
        {entry.resource_id && (
          <p className="text-sm text-muted-foreground">{entry.resource_id}</p>
        )}
      </div>
      <p className="shrink-0 text-xs text-muted-foreground" data-testid="entry-timestamp">
        {formatDate(entry.created_at)}
      </p>
    </>
  );

  return (
    <li className="border-b last:border-0">
      {isClickable ? (
        <button
          type="button"
          onClick={() => onNavigate(entry.resource_id!)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              onNavigate(entry.resource_id!);
            }
          }}
          className="flex w-full items-start justify-between gap-4 py-3 hover:bg-muted/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring text-left"
          data-testid={`exp-entry-${entry.id}`}
        >
          {content}
        </button>
      ) : (
        <div
          className="flex w-full items-start justify-between gap-4 py-3"
          data-testid={`exp-entry-${entry.id}`}
        >
          {content}
        </div>
      )}
    </li>
  );
}

// ---------------------------------------------------------------------------
// MyExperiments page
// ---------------------------------------------------------------------------

/**
 * My Experiments page — a filtered history view showing only OrcaLab
 * experiment activity.
 *
 * Fetches `GET /history/experiments` (service=orcalab filter applied
 * server-side) and displays entries in a list with status badges derived
 * from the action string. Clicking an entry navigates to the corresponding
 * OrcaLab experiment detail page.
 */
export function MyExperiments() {
  const navigate = useNavigate();

  const {
    data,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["history-experiments"],
    queryFn: async () => {
      const res = await apiClient.get<PaginatedResponse<ActivityLogEntry>>("/history/experiments");
      return res.data;
    },
    staleTime: 30_000,
  });

  const entries = data?.items ?? [];

  /** Navigate to the OrcaLab experiment detail page for the given experiment ID. */
  function handleNavigate(experimentId: string) {
    navigate(`${ROUTES.ORCALAB_EXPERIMENTS}/${experimentId}`);
  }

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-bold">My Experiments</h1>
        <p className="mt-1 text-muted-foreground">
          Your experiment activity in OrcaLab.
        </p>
      </div>

      {isLoading ? (
        <p className="text-muted-foreground" data-testid="my-exp-loading">
          Loading…
        </p>
      ) : isError ? (
        <p className="text-destructive" data-testid="my-exp-error">
          Failed to load experiment activity.
        </p>
      ) : entries.length === 0 ? (
        <p className="text-muted-foreground" data-testid="my-exp-empty">
          No experiment activity yet.
        </p>
      ) : (
        <ul data-testid="my-exp-list">
          {entries.map((entry) => (
            <ExperimentEntryRow key={entry.id} entry={entry} onNavigate={handleNavigate} />
          ))}
        </ul>
      )}
    </div>
  );
}
