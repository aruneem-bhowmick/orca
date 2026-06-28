import { useEffect, useRef, useState } from "react";
import { useInfiniteQuery } from "@tanstack/react-query";
import apiClient from "@/api/client";
import { formatDate } from "@/lib/utils";
import type { ActivityLogEntry, PaginatedResponse } from "@/api/types";

// ---------------------------------------------------------------------------
// Service badge
// ---------------------------------------------------------------------------

/** Colour classes for each service badge. */
const SERVICE_COLORS: Record<string, string> = {
  orcamind: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  orcalab: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
  orcanet: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200",
};

/**
 * Pill badge showing which Orca service generated an activity log entry.
 *
 * Falls back to a neutral muted style for unrecognised service values.
 *
 * @param props.service - The service identifier (e.g. "orcamind").
 */
function ServiceBadge({ service }: { service: string | null }) {
  if (!service) return null;
  const colorClass = SERVICE_COLORS[service] ?? "bg-muted text-muted-foreground";
  return (
    <span
      className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${colorClass}`}
      data-testid={`service-badge-${service}`}
    >
      {service}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Timeline entry
// ---------------------------------------------------------------------------

/**
 * A single activity log entry rendered as a timeline item.
 *
 * Shows the action label, resource type + ID (when present), a service
 * badge, and a human-readable timestamp.
 *
 * @param props.entry - The activity log entry to display.
 */
function TimelineEntry({ entry }: { entry: ActivityLogEntry }) {
  return (
    <li
      className="relative pl-6 pb-6 last:pb-0"
      data-testid={`activity-entry-${entry.id}`}
    >
      {/* Vertical line */}
      <span
        className="absolute left-0 top-1 flex h-4 w-4 -translate-x-[3px] items-center justify-center rounded-full border-2 border-primary bg-background"
        aria-hidden="true"
      />
      <span className="absolute left-0 top-5 bottom-0 w-px bg-border last:hidden" aria-hidden="true" />

      <div className="space-y-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-medium" data-testid="entry-action">
            {entry.action}
          </span>
          <ServiceBadge service={entry.service} />
        </div>
        {(entry.resource_type || entry.resource_id) && (
          <p className="text-sm text-muted-foreground">
            {entry.resource_type}
            {entry.resource_id ? `: ${entry.resource_id}` : ""}
          </p>
        )}
        <p className="text-xs text-muted-foreground" data-testid="entry-timestamp">
          {formatDate(entry.created_at)}
        </p>
      </div>
    </li>
  );
}

// ---------------------------------------------------------------------------
// Filter types
// ---------------------------------------------------------------------------

/** All valid service filter values including the "all" sentinel. */
const SERVICE_FILTERS = ["all", "orcamind", "orcalab", "orcanet"] as const;
type ServiceFilter = (typeof SERVICE_FILTERS)[number];

// ---------------------------------------------------------------------------
// ActivityLog page
// ---------------------------------------------------------------------------

/**
 * Activity Log page displaying the current user's full activity timeline.
 *
 * Data is fetched from `GET /history` with cursor-style infinite scroll:
 * as the user scrolls to the bottom of the list, additional pages are
 * loaded automatically via an `IntersectionObserver` watching a sentinel
 * element. The service filter dropdown narrows entries to a single
 * upstream service. Date range inputs further constrain the visible window
 * (client-side filter applied to the loaded pages).
 */
export function ActivityLog() {
  const [serviceFilter, setServiceFilter] = useState<ServiceFilter>("all");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const sentinelRef = useRef<HTMLDivElement | null>(null);

  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
    isError,
  } = useInfiniteQuery({
    queryKey: ["history-log", serviceFilter],
    queryFn: async ({ pageParam = 1 }) => {
      const params: Record<string, string | number> = {
        page: pageParam as number,
        per_page: 20,
      };
      if (serviceFilter !== "all") params.service = serviceFilter;
      const res = await apiClient.get<PaginatedResponse<ActivityLogEntry>>("/history", { params });
      return res.data;
    },
    getNextPageParam: (lastPage) =>
      lastPage.page < lastPage.pages ? lastPage.page + 1 : undefined,
    initialPageParam: 1,
  });

  // Auto-fetch next page when the sentinel scrolls into view.
  useEffect(() => {
    const el = sentinelRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasNextPage && !isFetchingNextPage) {
          fetchNextPage();
        }
      },
      { threshold: 0.1 },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [fetchNextPage, hasNextPage, isFetchingNextPage]);

  /** All entries across all loaded pages, flattened into a single array. */
  const allEntries = data?.pages.flatMap((p) => p.items) ?? [];

  /** Client-side date filter applied on top of the paginated server data. */
  const filtered = allEntries.filter((entry) => {
    const ts = new Date(entry.created_at).getTime();
    if (dateFrom && ts < new Date(dateFrom).getTime()) return false;
    if (dateTo && ts > new Date(dateTo + "T23:59:59Z").getTime()) return false;
    return true;
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Activity Log</h1>
        <p className="mt-1 text-muted-foreground">
          A timeline of your activity across all Orca services.
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-end gap-4">
        <div className="space-y-1">
          <label htmlFor="service-filter" className="text-sm font-medium">
            Service
          </label>
          <select
            id="service-filter"
            value={serviceFilter}
            onChange={(e) => setServiceFilter(e.target.value as ServiceFilter)}
            className="rounded-md border border-input bg-background px-3 py-1.5 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            data-testid="service-filter"
          >
            {SERVICE_FILTERS.map((s) => (
              <option key={s} value={s}>
                {s === "all" ? "All services" : s}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-1">
          <label htmlFor="date-from" className="text-sm font-medium">
            From
          </label>
          <input
            id="date-from"
            type="date"
            value={dateFrom}
            onChange={(e) => setDateFrom(e.target.value)}
            className="rounded-md border border-input bg-background px-3 py-1.5 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            data-testid="date-from"
          />
        </div>

        <div className="space-y-1">
          <label htmlFor="date-to" className="text-sm font-medium">
            To
          </label>
          <input
            id="date-to"
            type="date"
            value={dateTo}
            onChange={(e) => setDateTo(e.target.value)}
            className="rounded-md border border-input bg-background px-3 py-1.5 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            data-testid="date-to"
          />
        </div>
      </div>

      {/* Timeline */}
      {isLoading ? (
        <p className="text-muted-foreground" data-testid="activity-loading">
          Loading activity…
        </p>
      ) : isError ? (
        <p className="text-destructive" data-testid="activity-error">
          Failed to load activity log.
        </p>
      ) : filtered.length === 0 ? (
        <p className="text-muted-foreground" data-testid="activity-empty">
          No activity found.
        </p>
      ) : (
        <ul className="border-l-0" data-testid="activity-timeline">
          {filtered.map((entry) => (
            <TimelineEntry key={entry.id} entry={entry} />
          ))}
        </ul>
      )}

      {/* Infinite scroll sentinel */}
      <div ref={sentinelRef} data-testid="scroll-sentinel" />
      {isFetchingNextPage && (
        <p className="text-center text-sm text-muted-foreground" data-testid="loading-more">
          Loading more…
        </p>
      )}
      {!hasNextPage && filtered.length > 0 && (
        <p className="text-center text-sm text-muted-foreground" data-testid="end-of-log">
          End of activity log
        </p>
      )}
    </div>
  );
}
