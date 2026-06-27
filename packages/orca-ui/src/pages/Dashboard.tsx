import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import apiClient from "@/api/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { formatDate } from "@/lib/utils";
import { ROUTES } from "@/lib/constants";
import type {
  DashboardOverview,
  ActivityLogEntry,
  PaginatedResponse,
} from "@/api/types";

/**
 * A single summary stat card showing a metric label and its numeric value.
 *
 * @param props.title - Human-readable metric label.
 * @param props.value - The current numeric value, or undefined while loading.
 * @param props.testId - `data-testid` attribute applied to the value element.
 */
function StatCard({
  title,
  value,
  testId,
}: {
  title: string;
  value: number | undefined;
  testId: string;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-3xl font-bold" data-testid={testId}>
          {value ?? "—"}
        </p>
      </CardContent>
    </Card>
  );
}

/**
 * Main dashboard overview page shown after login.
 *
 * Fetches aggregated platform statistics from `GET /dashboard/overview` and
 * the ten most-recent activity log entries from `GET /history`. Renders four
 * summary stat cards, a quick-action button row for common workflows, and a
 * chronological activity timeline.
 */
export function Dashboard() {
  const navigate = useNavigate();

  const { data: overview, isLoading: overviewLoading, isError: overviewError } = useQuery({
    queryKey: ["dashboard-overview"],
    queryFn: async () => {
      const res = await apiClient.get<DashboardOverview>("/dashboard/overview");
      return res.data;
    },
  });

  const { data: activity, isLoading: activityLoading, isError: activityError } = useQuery({
    queryKey: ["history", { per_page: 10 }],
    queryFn: async () => {
      const res = await apiClient.get<PaginatedResponse<ActivityLogEntry>>(
        "/history",
        { params: { per_page: 10 } },
      );
      return res.data;
    },
  });

  return (
    <div>
      <h1 className="text-2xl font-bold">Dashboard</h1>
      <p className="mt-1 text-muted-foreground">Overview of the Orca platform.</p>

      {/* Summary stat cards */}
      <div
        className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4"
        data-testid="summary-cards"
      >
        {overviewLoading ? (
          <p className="col-span-4 text-muted-foreground">Loading stats…</p>
        ) : overviewError ? (
          <p
            className="col-span-4 text-destructive"
            data-testid="overview-error"
          >
            Failed to load overview stats.
          </p>
        ) : (
          <>
            <StatCard
              title="Total Tasks"
              value={overview?.total_tasks}
              testId="stat-total-tasks"
            />
            <StatCard
              title="Running Experiments"
              value={overview?.running_experiments}
              testId="stat-running-experiments"
            />
            <StatCard
              title="Completed Experiments"
              value={overview?.completed_experiments}
              testId="stat-completed-experiments"
            />
            <StatCard
              title="Recent Transfers"
              value={overview?.recent_transfers}
              testId="stat-recent-transfers"
            />
          </>
        )}
      </div>

      {/* Quick actions */}
      <div className="mt-8">
        <h2 className="text-lg font-semibold">Quick Actions</h2>
        <div
          className="mt-3 flex flex-wrap gap-3"
          data-testid="quick-actions"
        >
          <Button onClick={() => navigate(ROUTES.ORCAMIND_TASKS)}>
            New Task
          </Button>
          <Button
            variant="secondary"
            onClick={() => navigate(ROUTES.ORCALAB_EXPERIMENTS)}
          >
            Start Experiment
          </Button>
          <Button
            variant="outline"
            onClick={() => navigate(ROUTES.ORCANET_TRANSFER)}
          >
            Score Transfer
          </Button>
        </div>
      </div>

      {/* Recent activity timeline */}
      <div className="mt-8">
        <h2 className="text-lg font-semibold">Recent Activity</h2>
        {activityLoading ? (
          <p className="mt-2 text-muted-foreground">Loading activity…</p>
        ) : activityError ? (
          <p
            className="mt-2 text-destructive"
            data-testid="activity-error"
          >
            Failed to load recent activity.
          </p>
        ) : !activity?.items.length ? (
          <p
            className="mt-2 text-muted-foreground"
            data-testid="no-activity"
          >
            No recent activity.
          </p>
        ) : (
          <ol
            className="mt-3 space-y-3 border-l-2 border-border pl-4"
            data-testid="activity-timeline"
          >
            {activity.items.map((entry) => (
              <li key={entry.id} className="flex flex-col gap-0.5 text-sm">
                <div className="flex items-center gap-2">
                  <span className="font-medium">{entry.action}</span>
                  {entry.service && (
                    <span
                      className="rounded-full bg-secondary px-2 py-0.5 text-xs"
                      data-testid={`activity-service-${entry.id}`}
                    >
                      {entry.service}
                    </span>
                  )}
                  {entry.resource_type && (
                    <span className="text-muted-foreground">
                      {entry.resource_type}
                    </span>
                  )}
                </div>
                <span className="text-xs text-muted-foreground">
                  {formatDate(entry.created_at)}
                </span>
              </li>
            ))}
          </ol>
        )}
      </div>
    </div>
  );
}
