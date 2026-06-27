/**
 * Utility functions for the orca-ui frontend.
 *
 * @module lib/utils
 */
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merge Tailwind CSS class names with conflict resolution.
 *
 * Combines `clsx` (conditional class joining) with `tailwind-merge`
 * (deduplication of conflicting Tailwind utilities). Use this wherever
 * component class names need to be composed from multiple sources.
 *
 * @param inputs - Class values: strings, arrays, objects, or falsy values.
 * @returns A single deduplicated class name string.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Regex pattern for basic email format validation (user@domain.tld). */
export const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

/**
 * Format an ISO 8601 date string into a human-readable form.
 *
 * Returns the original string if the input is not a valid date.
 *
 * @param dateString - An ISO date string (e.g. "2024-01-15T12:00:00Z").
 * @returns Formatted string in "Month DD, YYYY HH:MM" format.
 */
export function formatDate(dateString: string): string {
  const date = new Date(dateString);
  if (isNaN(date.getTime())) {
    return dateString;
  }
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

/**
 * Calculate the elapsed duration between a start time and an optional
 * end time (defaults to now if `endDateString` is `null`).
 *
 * Returns a human-readable string such as `"2h 15m"`, `"45m 3s"`, or
 * `"8s"`. Returns `"—"` if `startDateString` is falsy.
 *
 * @param startDateString - ISO 8601 start timestamp.
 * @param endDateString - ISO 8601 end timestamp, or `null` to use now.
 * @returns Human-readable elapsed time string.
 */
export function formatElapsed(
  startDateString: string,
  endDateString: string | null = null,
): string {
  const start = new Date(startDateString);
  if (isNaN(start.getTime())) return "—";

  const end = endDateString ? new Date(endDateString) : new Date();
  const totalSeconds = Math.max(0, Math.floor((end.getTime() - start.getTime()) / 1_000));

  const hours = Math.floor(totalSeconds / 3_600);
  const minutes = Math.floor((totalSeconds % 3_600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}
