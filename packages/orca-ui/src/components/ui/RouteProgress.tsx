/**
 * Thin progress bar that appears at the top of the viewport during route
 * transitions, giving users a visual cue that navigation is in progress.
 *
 * The bar slides from 0 to ~80% of the viewport width during the transition
 * and completes to 100% once the new route renders, then fades out.
 *
 * Implementation uses React Router's `useNavigation` hook (v6.4+) to detect
 * the pending navigation state.
 *
 * @module components/ui/RouteProgress
 */
import { useEffect, useState } from "react";
import { useNavigation } from "react-router-dom";

/** Duration of the fade-out animation in milliseconds. */
const FADE_DURATION_MS = 300;

/**
 * Route-transition progress bar.
 *
 * Mount this inside any component that is already within a `BrowserRouter`
 * context (e.g. at the top of `App` before any routes, or inside
 * `MainLayout`). It renders `null` when no navigation is pending.
 */
export function RouteProgress() {
  const navigation = useNavigation();
  const isNavigating = navigation.state !== "idle";

  const [progress, setProgress] = useState(0);
  const [visible, setVisible] = useState(false);
  const [fading, setFading] = useState(false);

  useEffect(() => {
    let rafId: number;
    let intervalId: ReturnType<typeof setInterval>;

    if (isNavigating) {
      setFading(false);
      setVisible(true);
      setProgress(0);

      // Ramp up to ~80% smoothly while navigation is pending.
      let current = 0;
      intervalId = setInterval(() => {
        current = Math.min(current + Math.random() * 12 + 3, 80);
        setProgress(current);
      }, 80);
    } else if (visible) {
      // Navigation complete — complete the bar and fade out.
      setProgress(100);
      rafId = requestAnimationFrame(() => {
        setTimeout(() => {
          setFading(true);
          setTimeout(() => {
            setVisible(false);
            setFading(false);
            setProgress(0);
          }, FADE_DURATION_MS);
        }, 100);
      });
    }

    return () => {
      clearInterval(intervalId);
      cancelAnimationFrame(rafId);
    };
  // Depend only on isNavigating to avoid stale closure traps.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isNavigating]);

  if (!visible) return null;

  return (
    <div
      className="pointer-events-none fixed left-0 top-0 z-[200] h-1 w-full"
      aria-hidden="true"
      data-testid="route-progress"
    >
      <div
        className="h-full bg-primary transition-[width] duration-200 ease-out"
        style={{
          width: `${progress}%`,
          opacity: fading ? 0 : 1,
          transition: fading
            ? `opacity ${FADE_DURATION_MS}ms ease-out`
            : "width 200ms ease-out",
        }}
      />
    </div>
  );
}
