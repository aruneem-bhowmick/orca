/**
 * Skeleton loader component for content that is still loading.
 *
 * Renders a pulsing placeholder block styled with the current theme's muted
 * colour. Use it to replace tables, cards, and other data-heavy sections
 * while an API request is in flight, giving users instant visual feedback.
 *
 * @module components/ui/Skeleton
 */
import { cn } from "@/lib/utils";
import type { HTMLAttributes } from "react";

/**
 * A single skeleton placeholder block.
 *
 * Accepts all standard `<div>` props so callers can control width, height,
 * and border-radius via `className`.
 *
 * @param props.className - Additional Tailwind classes (e.g. `"h-4 w-24"`).
 *
 * @example
 * <Skeleton className="h-4 w-full" />
 * <Skeleton className="h-10 w-32 rounded-full" />
 */
export function Skeleton({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("animate-pulse rounded-md bg-muted", className)}
      {...props}
    />
  );
}

/**
 * A pre-composed skeleton that mimics a data table with configurable rows.
 *
 * Renders a header row followed by `rows` body rows, each containing
 * five equally-spaced column cells.
 *
 * @param props.rows - Number of body skeleton rows to render (default 5).
 */
export function TableSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div className="overflow-auto rounded-lg border" data-testid="table-skeleton">
      {/* Header row */}
      <div className="border-b bg-muted/50 px-4 py-3">
        <div className="flex gap-4">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-4 flex-1" />
          ))}
        </div>
      </div>
      {/* Body rows */}
      {Array.from({ length: rows }).map((_, row) => (
        <div key={row} className="border-b px-4 py-3 last:border-0">
          <div className="flex gap-4">
            {Array.from({ length: 5 }).map((_, col) => (
              <Skeleton key={col} className="h-4 flex-1" />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * A pre-composed skeleton that mimics a row of summary stat cards.
 *
 * @param props.count - Number of card skeletons to render (default 4).
 */
export function CardRowSkeleton({ count = 4 }: { count?: number }) {
  return (
    <div
      className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4"
      data-testid="card-row-skeleton"
    >
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="rounded-lg border bg-card p-6 shadow-sm">
          <Skeleton className="mb-3 h-4 w-24" />
          <Skeleton className="h-8 w-16" />
        </div>
      ))}
    </div>
  );
}
