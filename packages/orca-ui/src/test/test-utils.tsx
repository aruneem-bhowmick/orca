/**
 * Test utilities for orca-ui component and hook tests.
 *
 * Re-exports everything from `@testing-library/react` and provides a
 * custom `render()` function that wraps components in the required
 * providers (QueryClientProvider, MemoryRouter). The QueryClient is
 * configured with no retries and zero GC time to ensure deterministic
 * test behaviour.
 *
 * @module test/test-utils
 */
import { type ReactElement } from "react";
import { render, type RenderOptions } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

/** Create a QueryClient configured for testing (no retries, no cache). */
function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });
}

/** Wrapper component that provides all required context providers for testing. */
function AllProviders({ children }: { children: React.ReactNode }) {
  const queryClient = createTestQueryClient();
  return (
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryClientProvider>
  );
}

/**
 * Render a React element wrapped in all providers required by orca-ui.
 *
 * @param ui - The React element to render.
 * @param options - Optional Testing Library render options.
 * @returns The Testing Library render result.
 */
function customRender(ui: ReactElement, options?: Omit<RenderOptions, "wrapper">) {
  return render(ui, { wrapper: AllProviders, ...options });
}

export * from "@testing-library/react";
export { customRender as render };
