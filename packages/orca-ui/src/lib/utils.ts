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
