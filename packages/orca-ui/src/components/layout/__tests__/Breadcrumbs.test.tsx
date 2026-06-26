import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import { render as rtlRender } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Header } from "@/components/layout/Header";
import { useAuthStore } from "@/store/auth";
import { mockUser } from "@/test/mocks/handlers";

vi.mock("@/api/auth", () => ({
  getMe: vi.fn().mockRejectedValue(new Error("No session")),
}));

/**
 * Render the Header at a specific initial route path to test
 * breadcrumb generation for different URL patterns.
 *
 * @param initialPath - The URL path to simulate (e.g. "/dashboard/orcamind/tasks").
 */
function renderAtPath(initialPath: string) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0, refetchOnWindowFocus: false },
    },
  });

  return rtlRender(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[initialPath]}>
        <Header />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("Header breadcrumbs", () => {
  beforeEach(() => {
    useAuthStore.getState().setAuth(mockUser, "test-token");
    document.documentElement.classList.remove("dark");
  });

  it("shows 'Home' when at the root path", () => {
    renderAtPath("/");
    expect(screen.getByText("Home")).toBeInTheDocument();
  });

  it("shows 'Dashboard' for /dashboard", () => {
    renderAtPath("/dashboard");
    expect(screen.getByText("Dashboard")).toBeInTheDocument();
  });

  it("shows hierarchical breadcrumbs for nested routes", () => {
    renderAtPath("/dashboard/orcamind/tasks");
    expect(screen.getByText("Dashboard")).toBeInTheDocument();
    expect(screen.getByText("OrcaMind")).toBeInTheDocument();
    expect(screen.getByText("Tasks")).toBeInTheDocument();
  });

  it("renders intermediate breadcrumbs as links", () => {
    renderAtPath("/dashboard/orcalab/experiments");
    const dashboardLink = screen.getByText("Dashboard");
    expect(dashboardLink.tagName).toBe("A");
    expect(dashboardLink).toHaveAttribute("href", "/dashboard");
  });

  it("renders the last breadcrumb as plain text (not a link)", () => {
    renderAtPath("/dashboard/orcalab/experiments");
    const experimentsText = screen.getByText("Experiments");
    expect(experimentsText.tagName).not.toBe("A");
    expect(experimentsText.tagName).toBe("SPAN");
  });

  it("shows separator between breadcrumb segments", () => {
    renderAtPath("/dashboard/orcamind/tasks");
    const separators = screen.getAllByText("/");
    expect(separators.length).toBe(2);
  });

  it("uses human-readable labels for known path segments", () => {
    renderAtPath("/history/tasks");
    expect(screen.getByText("History")).toBeInTheDocument();
    expect(screen.getByText("Tasks")).toBeInTheDocument();
  });

  it("capitalises unknown path segments", () => {
    renderAtPath("/dashboard/unknown-page");
    expect(screen.getByText("Unknown-page")).toBeInTheDocument();
  });

  it("renders the breadcrumb nav with aria-label", () => {
    renderAtPath("/dashboard");
    const nav = screen.getByLabelText("Breadcrumb");
    expect(nav).toBeInTheDocument();
  });
});
