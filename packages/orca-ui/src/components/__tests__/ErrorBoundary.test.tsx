import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ErrorBoundary } from "@/components/ErrorBoundary";

/** A component that throws an error during render for testing purposes. */
function ThrowingChild({ shouldThrow }: { shouldThrow: boolean }) {
  if (shouldThrow) throw new Error("Test render error");
  return <div data-testid="child-ok">All good</div>;
}

describe("ErrorBoundary", () => {
  beforeEach(() => {
    // Suppress expected React error boundary console output in tests.
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders children when no error is thrown", () => {
    render(
      <ErrorBoundary>
        <ThrowingChild shouldThrow={false} />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId("child-ok")).toBeInTheDocument();
  });

  it("shows the default fallback UI when a child throws", () => {
    render(
      <ErrorBoundary>
        <ThrowingChild shouldThrow={true} />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId("error-boundary-fallback")).toBeInTheDocument();
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
  });

  it("shows the Go Home link in the fallback", () => {
    render(
      <ErrorBoundary>
        <ThrowingChild shouldThrow={true} />
      </ErrorBoundary>,
    );
    const homeLink = screen.getByTestId("error-go-home");
    expect(homeLink).toBeInTheDocument();
    expect(homeLink).toHaveAttribute("href", "/");
  });

  it("shows a Try Again button in the fallback", () => {
    render(
      <ErrorBoundary>
        <ThrowingChild shouldThrow={true} />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId("error-try-again")).toBeInTheDocument();
  });

  it("renders a custom fallback when provided", () => {
    render(
      <ErrorBoundary fallback={(err) => <p>Custom: {err.message}</p>}>
        <ThrowingChild shouldThrow={true} />
      </ErrorBoundary>,
    );
    expect(screen.getByText("Custom: Test render error")).toBeInTheDocument();
    expect(screen.queryByTestId("error-boundary-fallback")).not.toBeInTheDocument();
  });

  it("recovers and re-renders children after clicking Try Again", () => {
    /**
     * Mutable holder used to control whether the child throws. We use a plain
     * object so that the reference stays stable across re-renders while still
     * being mutated from the test.
     */
    const state = { shouldThrow: true };

    function ToggleableChild() {
      if (state.shouldThrow) throw new Error("Transient error");
      return <div data-testid="recovered">Recovered</div>;
    }

    render(
      <ErrorBoundary>
        <ToggleableChild />
      </ErrorBoundary>,
    );

    expect(screen.getByTestId("error-boundary-fallback")).toBeInTheDocument();

    // Stop throwing on next render.
    state.shouldThrow = false;
    fireEvent.click(screen.getByTestId("error-try-again"));

    expect(screen.getByTestId("recovered")).toBeInTheDocument();
  });
});
