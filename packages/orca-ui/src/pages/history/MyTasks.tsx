import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import apiClient from "@/api/client";
import { formatDate } from "@/lib/utils";
import { ROUTES } from "@/lib/constants";
import type { ActivityLogEntry, PaginatedResponse } from "@/api/types";

// ---------------------------------------------------------------------------
// Task entry row
// ---------------------------------------------------------------------------

/**
 * A single task-related activity entry row, linking to the OrcaMind task
 * detail page for the referenced resource.
 *
 * @param props.entry - The activity log entry to render.
 * @param props.onNavigate - Callback invoked with the task ID when the row is clicked.
 */
function TaskEntryRow({
  entry,
  onNavigate,
}: {
  entry: ActivityLogEntry;
  onNavigate: (taskId: string) => void;
}) {
  const isClickable = !!entry.resource_id;

  return (
    <li
      className={`flex items-start justify-between gap-4 border-b py-3 last:border-0 ${isClickable ? "cursor-pointer hover:bg-muted/30" : ""}`}
      onClick={() => isClickable && onNavigate(entry.resource_id!)}
      onKeyDown={(e) => {
        if (isClickable && (e.key === "Enter" || e.key === " ")) {
          e.preventDefault();
          onNavigate(entry.resource_id!);
        }
      }}
      tabIndex={isClickable ? 0 : undefined}
      data-testid={`task-entry-${entry.id}`}
    >
      <div className="space-y-0.5">
        <p className="font-medium" data-testid="entry-action">
          {entry.action}
        </p>
        {entry.resource_id && (
          <p className="text-sm text-muted-foreground">{entry.resource_id}</p>
        )}
      </div>
      <p className="shrink-0 text-xs text-muted-foreground" data-testid="entry-timestamp">
        {formatDate(entry.created_at)}
      </p>
    </li>
  );
}

// ---------------------------------------------------------------------------
// MyTasks page
// ---------------------------------------------------------------------------

/**
 * My Tasks page — a filtered history view showing only OrcaMind task activity.
 *
 * Fetches `GET /history/tasks` (service=orcamind filter applied server-side)
 * and displays entries in a list. Clicking an entry navigates to the
 * corresponding OrcaMind task detail page.
 */
export function MyTasks() {
  const navigate = useNavigate();

  const {
    data,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["history-tasks"],
    queryFn: async () => {
      const res = await apiClient.get<PaginatedResponse<ActivityLogEntry>>("/history/tasks");
      return res.data;
    },
    staleTime: 30_000,
  });

  const entries = data?.items ?? [];

  /** Navigate to the OrcaMind task detail page for the given task ID. */
  function handleNavigate(taskId: string) {
    navigate(`${ROUTES.ORCAMIND_TASKS}/${taskId}`);
  }

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-bold">My Tasks</h1>
        <p className="mt-1 text-muted-foreground">
          Your task-related activity in OrcaMind.
        </p>
      </div>

      {isLoading ? (
        <p className="text-muted-foreground" data-testid="my-tasks-loading">
          Loading…
        </p>
      ) : isError ? (
        <p className="text-destructive" data-testid="my-tasks-error">
          Failed to load task activity.
        </p>
      ) : entries.length === 0 ? (
        <p className="text-muted-foreground" data-testid="my-tasks-empty">
          No task activity yet.
        </p>
      ) : (
        <ul data-testid="my-tasks-list">
          {entries.map((entry) => (
            <TaskEntryRow key={entry.id} entry={entry} onNavigate={handleNavigate} />
          ))}
        </ul>
      )}
    </div>
  );
}
