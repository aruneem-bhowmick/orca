/**
 * Thin progress bar that appears at the top of the viewport during route
 * transitions, giving users a visual cue that navigation is in progress.
 *
 * The bar slides from 0 to ~80% of the viewport width during the transition
 * and completes to 100% once the new route renders, then fades out.
 *
 * Implementation uses React Router's `useNavigation` hook (data-router API,
 * v6.4+). The component is split so that `RouteProgressBar` only mounts when
 * a data-router context is available; in legacy-router or test environments
 * where that context is absent the outer `RouteProgress` renders nothing.
 *
 * @module components/ui/RouteProgress
 */
import { useContext, useEffect, useState } from "react";
import { useNavigation, UNSAFE_DataRouterStateContext } from "react-router-dom";

/** Duration of the fade-out animation in milliseconds. */
const FADE_DURATION_MS = 300;

/**
 * Inner implementation of the progress bar.
 *
 * Only mounted when a data-router state context is present, so `useNavigation`
 * is always called in a valid context.
 */
function RouteProgressBar() {
  const navigation = useNavigation();
  const isNavigating = navigation.state !== "idle";

  const [progress, setProgress] = useState(0);
  const [visible, setVisible] = useState(false);
  const [fading, setFading] = useState(false);

  useEffect(() => {
    let rafId: number;
    let intervalId: ReturnType<typeof setInterval>;
    let timeoutId1: ReturnType<typeof setTimeout>;
    let timeoutId2: ReturnType<typeof setTimeout>;

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
        timeoutId1 = setTimeout(() => {
          setFading(true);
          timeoutId2 = setTimeout(() => {
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
      clearTimeout(timeoutId1);
      clearTimeout(timeoutId2);
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

/**
 * Route-transition progress bar.
 *
 * Mount this inside any component that is already within a router context.
 * When a data-router context is present (i.e. the app uses
 * `createBrowserRouter` / `RouterProvider`), the progress bar is active.
 * In legacy-router environments (`BrowserRouter`) the component renders
 * nothing, avoiding the invariant thrown by `useNavigation` outside a
 * data-router context.
 */
export function RouteProgress() {
  const routerState = useContext(UNSAFE_DataRouterStateContext);
  if (!routerState) return null;
  return <RouteProgressBar />;
}
