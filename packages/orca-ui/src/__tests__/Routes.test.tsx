import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { App } from "@/App";
import { useAuthStore } from "@/store/auth";
import { mockUser } from "@/test/mocks/handlers";
import * as authApi from "@/api/auth";

vi.mock("@/api/auth", () => ({
  getMe: vi.fn(),
  logout: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("@/api/client", () => ({
  default: {
    get: vi.fn().mockImplementation((url: string) => {
      if (url === "/health") {
        return Promise.resolve({
          data: {
            status: "healthy",
            services: {
              postgres: true,
              redis: true,
              orcamind: true,
              orcalab: true,
              orcanet: true,
            },
          },
        });
      }
      if (url === "/dashboard/stats") {
        return Promise.resolve({
          data: {
            tasks_registered: 42,
            experiments_run: 128,
            transfers_scored: 56,
          },
        });
      }
      if (url === "/dashboard/overview") {
        return Promise.resolve({
          data: {
            total_tasks: 5,
            running_experiments: 2,
            completed_experiments: 10,
            recent_transfers: 3,
          },
        });
      }
      if (url === "/history") {
        return Promise.resolve({ data: { items: [], total: 0, page: 1, per_page: 10, pages: 0 } });
      }
      if (url === "/orcamind/tasks") {
        return Promise.resolve({ data: [] });
      }
      if (url.startsWith("/orcamind/tasks/")) {
        return Promise.resolve({
          data: {
            task_id: url.split("/").pop(),
            name: "Test Task",
            domain: "vision",
            task_type: "classification",
            n_samples: 1000,
            n_features: null,
            n_classes: 10,
            metadata: null,
            created_at: "2024-01-01T00:00:00Z",
          },
        });
      }
      return Promise.reject(new Error(`Unexpected API call: GET ${url}`));
    }),
    interceptors: {
      request: { use: vi.fn(), handlers: [] },
      response: { use: vi.fn(), handlers: [] },
    },
  },
}));

/**
 * Render the full App at a given path with the user authenticated.
 * Mocks getMe to resolve with the mockUser so the useAuth hook
 * completes its session restoration successfully.
 *
 * @param path - The URL path to navigate to.
 */
function renderAuthenticatedAt(path: string) {
  useAuthStore.getState().setAuth(mockUser, "test-token");
  vi.mocked(authApi.getMe).mockResolvedValue(mockUser);
  window.history.pushState({}, "", path);
  return render(<App />);
}

/**
 * Find the heading (h1) rendered by PlaceholderPage in the main content area.
 * This distinguishes the page heading from sidebar navigation labels that may
 * have the same text.
 *
 * @param text - The expected heading text.
 */
function findPageHeading(text: string): HTMLElement | null {
  const headings = screen.getAllByRole("heading", { level: 1 });
  return headings.find((h) => h.textContent === text) ?? null;
}

describe("Protected route map", () => {
  beforeEach(() => {
    useAuthStore.getState().clearAuth();
    vi.clearAllMocks();
  });

  it("redirects unauthenticated users from /dashboard to /login", async () => {
    vi.mocked(authApi.getMe).mockRejectedValue(new Error("No session"));
    window.history.pushState({}, "", "/dashboard");
    render(<App />);
    expect(screen.getByTestId("auth-loading")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByText("Sign in to Orca")).toBeInTheDocument();
    });
  });

  it("renders the OrcaMind Tasks page at /dashboard/orcamind/tasks", async () => {
    renderAuthenticatedAt("/dashboard/orcamind/tasks");
    await waitFor(() => {
      expect(findPageHeading("Tasks")).toBeInTheDocument();
    });
  });

  it("renders the OrcaLab Experiments page at /dashboard/orcalab/experiments", async () => {
    renderAuthenticatedAt("/dashboard/orcalab/experiments");
    await waitFor(() => {
      expect(findPageHeading("Experiments")).toBeInTheDocument();
    });
  });

  it("renders the OrcaLab Sweeps page at /dashboard/orcalab/sweeps", async () => {
    renderAuthenticatedAt("/dashboard/orcalab/sweeps");
    await waitFor(() => {
      expect(findPageHeading("Sweeps")).toBeInTheDocument();
    });
  });

  it("renders the OrcaNet Transfer page at /dashboard/orcanet/transfer", async () => {
    renderAuthenticatedAt("/dashboard/orcanet/transfer");
    await waitFor(() => {
      expect(findPageHeading("Transfer Explorer")).toBeInTheDocument();
    });
  });

  it("renders the OrcaNet Retrieval page at /dashboard/orcanet/retrieve", async () => {
    renderAuthenticatedAt("/dashboard/orcanet/retrieve");
    await waitFor(() => {
      expect(findPageHeading("Retrieval")).toBeInTheDocument();
    });
  });

  it("renders the Activity Log page at /history", async () => {
    renderAuthenticatedAt("/history");
    await waitFor(() => {
      expect(findPageHeading("Activity Log")).toBeInTheDocument();
    });
  });

  it("renders the My Tasks page at /history/tasks", async () => {
    renderAuthenticatedAt("/history/tasks");
    await waitFor(() => {
      expect(findPageHeading("My Tasks")).toBeInTheDocument();
    });
  });

  it("renders the My Experiments page at /history/experiments", async () => {
    renderAuthenticatedAt("/history/experiments");
    await waitFor(() => {
      expect(findPageHeading("My Experiments")).toBeInTheDocument();
    });
  });

  it("renders the Bookmarks page at /bookmarks", async () => {
    renderAuthenticatedAt("/bookmarks");
    await waitFor(() => {
      expect(findPageHeading("Bookmarks")).toBeInTheDocument();
    });
  });

  it("renders the Profile page at /profile", async () => {
    renderAuthenticatedAt("/profile");
    await waitFor(() => {
      expect(findPageHeading("Profile")).toBeInTheDocument();
    });
  });

  it("renders the Task Detail page at /dashboard/orcamind/tasks/:id", async () => {
    renderAuthenticatedAt("/dashboard/orcamind/tasks/abc-123");
    await waitFor(() => {
      expect(findPageHeading("Task Detail")).toBeInTheDocument();
    });
  });

  it("renders the Experiment Detail page at /dashboard/orcalab/experiments/:id", async () => {
    renderAuthenticatedAt("/dashboard/orcalab/experiments/exp-456");
    await waitFor(() => {
      expect(findPageHeading("Experiment Detail")).toBeInTheDocument();
    });
  });
});
