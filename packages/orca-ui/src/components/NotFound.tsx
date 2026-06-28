/**
 * 404 Not Found page component.
 *
 * Rendered when the user navigates to a route that does not exist in the
 * application's route tree. Provides a clear message and a "Go Home" link
 * so the user can recover without using the browser's back button.
 *
 * @module components/NotFound
 */
import { Link } from "react-router-dom";
import { ROUTES } from "@/lib/constants";

/**
 * 404 Not Found page.
 *
 * Displays a full-page centred layout with a large "404" heading,
 * a descriptive message, and a link back to the dashboard home.
 */
export function NotFound() {
  return (
    <div
      className="flex min-h-screen items-center justify-center p-8"
      data-testid="not-found-page"
    >
      <div className="text-center">
        <p className="text-6xl font-extrabold text-primary" aria-hidden="true">
          404
        </p>
        <h1 className="mt-4 text-2xl font-semibold">Page not found</h1>
        <p className="mt-2 text-muted-foreground">
          The page you are looking for does not exist or has been moved.
        </p>
        <div className="mt-8">
          <Link
            to={ROUTES.HOME}
            className="inline-flex items-center justify-center rounded-md bg-primary px-6 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            data-testid="not-found-go-home"
          >
            Go Home
          </Link>
        </div>
      </div>
    </div>
  );
}
