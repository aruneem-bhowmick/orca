/**
 * Spinner and overlay components for indicating in-progress operations.
 *
 * `Spinner` is a small inline SVG animation suitable for use inside buttons or
 * inline loading indicators. `SpinnerOverlay` renders a full-container
 * semi-transparent layer with a centred spinner, used to disable form
 * interaction during submission.
 *
 * @module components/ui/Spinner
 */
import { cn } from "@/lib/utils";

/**
 * Animated SVG spinner icon.
 *
 * @param props.className - Additional Tailwind classes. Controls size via
 *   `h-*`/`w-*` utilities (default `h-5 w-5`).
 * @param props.label - Accessible label for screen readers (default
 *   `"Loading"`).
 */
export function Spinner({
  className,
  label = "Loading",
}: {
  className?: string;
  label?: string;
}) {
  return (
    <svg
      className={cn("animate-spin text-primary", className ?? "h-5 w-5")}
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      aria-label={label}
      role="status"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}

/**
 * Full-container spinner overlay.
 *
 * Renders a translucent backdrop that covers its nearest `position: relative`
 * ancestor and shows a centred spinner. Prevents user interaction with
 * underlying content while a form submission or async operation is in flight.
 *
 * @param props.label - Accessible label forwarded to the inner `Spinner`.
 */
export function SpinnerOverlay({ label = "Loading" }: { label?: string }) {
  return (
    <div
      className="absolute inset-0 z-10 flex items-center justify-center rounded-md bg-background/60 backdrop-blur-sm"
      data-testid="spinner-overlay"
    >
      <Spinner className="h-8 w-8" label={label} />
    </div>
  );
}
