/**
 * React class-based error boundary that catches unhandled rendering errors.
 *
 * When any descendant component throws during render, `componentDidUpdate`, or
 * `getDerivedStateFromError`, `ErrorBoundary` replaces the subtree with a
 * user-friendly "Something went wrong" fallback UI. The user can navigate back
 * to the home page without a full browser reload.
 *
 * Use by wrapping the main layout or any subtree that should be isolated:
 *
 * ```tsx
 * <ErrorBoundary>
 *   <MainLayout />
 * </ErrorBoundary>
 * ```
 *
 * @module components/ErrorBoundary
 */
import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  /** The subtree to protect. */
  children: ReactNode;
  /**
   * Optional custom fallback rendered instead of the default error screen.
   * Receives the caught error for display.
   */
  fallback?: (error: Error) => ReactNode;
}

interface State {
  /** Whether an error has been caught. */
  hasError: boolean;
  /** The caught error, available to render in the fallback. */
  error: Error | null;
}

/**
 * Error boundary that wraps a subtree and displays a recovery UI on failure.
 *
 * Renders `props.fallback(error)` when provided, otherwise shows a default
 * "Something went wrong" card with a "Go Home" button.
 */
export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[ErrorBoundary] Caught rendering error:", error, info);
  }

  /**
   * Reset the error state so the user can attempt to reload the failed subtree.
   */
  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    const { hasError, error } = this.state;
    const { children, fallback } = this.props;

    if (!hasError) return children;

    if (fallback && error) return fallback(error);

    return (
      <div
        className="flex min-h-screen items-center justify-center p-8"
        data-testid="error-boundary-fallback"
      >
        <div className="w-full max-w-md rounded-lg border bg-card p-8 shadow-sm text-center">
          <div className="mb-4 flex justify-center">
            <svg
              className="h-12 w-12 text-destructive"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
              />
            </svg>
          </div>
          <h1 className="mb-2 text-xl font-semibold">Something went wrong</h1>
          <p className="mb-6 text-sm text-muted-foreground">
            An unexpected error occurred. You can go back to the home page or
            try again.
          </p>
          {error && (
            <details className="mb-6 text-left">
              <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground">
                Error details
              </summary>
              <pre className="mt-2 overflow-auto rounded-md bg-muted p-3 text-xs text-muted-foreground">
                {error.message}
              </pre>
            </details>
          )}
          <div className="flex justify-center gap-3">
            <a
              href="/"
              className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              data-testid="error-go-home"
            >
              Go Home
            </a>
            <button
              onClick={this.handleReset}
              className="inline-flex items-center justify-center rounded-md border px-4 py-2 text-sm font-medium hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              data-testid="error-try-again"
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }
}
