import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { Dashboard } from "@/pages/Dashboard";
import apiClient from "@/api/client";
import {
  mockDashboardOverview,
  mockActivityPage,
} from "@/test/mocks/handlers";

vi.mock("@/api/client", () => ({
  default: {
    get: vi.fn(),
    interceptors: {
      request: { use: vi.fn(), handlers: [] },
      response: { use: vi.fn(), handlers: [] },
    },
  },
}));

/** Module-level navigate spy so the useNavigate mock can reference it. */
const mockNavigate = vi.fn();

vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>();
  return { ...actual, useNavigate: () => mockNavigate };
});

describe("Dashboard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(apiClient.get).mockImplementation((url: string) => {
      if (url === "/dashboard/overview") {
        return Promise.resolve({ data: mockDashboardOverview });
      }
      if (url === "/history") {
        return Promise.resolve({ data: mockActivityPage });
      }
      return Promise.reject(new Error(`Unexpected: ${url}`));
    });
  });

  it("renders the page heading", () => {
    render(<Dashboard />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "Dashboard",
    );
  });

  it("renders the summary cards container", () => {
    render(<Dashboard />);
    expect(screen.getByTestId("summary-cards")).toBeInTheDocument();
  });

  it("populates summary stat cards from overview data", async () => {
    render(<Dashboard />);
    await waitFor(() => {
      expect(screen.getByTestId("stat-total-tasks")).toHaveTextContent("12");
    });
    expect(screen.getByTestId("stat-running-experiments")).toHaveTextContent(
      "3",
    );
    expect(
      screen.getByTestId("stat-completed-experiments"),
    ).toHaveTextContent("27");
    expect(screen.getByTestId("stat-recent-transfers")).toHaveTextContent("5");
  });

  it("renders all four stat card titles", async () => {
    render(<Dashboard />);
    await waitFor(() => {
      expect(screen.getByText("Total Tasks")).toBeInTheDocument();
    });
    expect(screen.getByText("Running Experiments")).toBeInTheDocument();
    expect(screen.getByText("Completed Experiments")).toBeInTheDocument();
    expect(screen.getByText("Recent Transfers")).toBeInTheDocument();
  });

  it("renders the quick-actions section with three buttons", () => {
    render(<Dashboard />);
    const actions = screen.getByTestId("quick-actions");
    expect(actions).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /new task/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /start experiment/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /score transfer/i }),
    ).toBeInTheDocument();
  });

  it("renders the recent activity timeline when entries exist", async () => {
    render(<Dashboard />);
    await waitFor(() => {
      expect(screen.getByTestId("activity-timeline")).toBeInTheDocument();
    });
    expect(screen.getByText("task_created")).toBeInTheDocument();
  });

  it("shows a service badge on activity entries that have a service", async () => {
    render(<Dashboard />);
    await waitFor(() => {
      expect(
        screen.getByTestId("activity-service-log-001"),
      ).toHaveTextContent("orcamind");
    });
  });

  it("shows the no-activity message when the history is empty", async () => {
    vi.mocked(apiClient.get).mockImplementation((url: string) => {
      if (url === "/dashboard/overview") {
        return Promise.resolve({ data: mockDashboardOverview });
      }
      if (url === "/history") {
        return Promise.resolve({
          data: { items: [], total: 0, page: 1, per_page: 10, pages: 0 },
        });
      }
      return Promise.reject(new Error(`Unexpected: ${url}`));
    });
    render(<Dashboard />);
    await waitFor(() => {
      expect(screen.getByTestId("no-activity")).toBeInTheDocument();
    });
  });

  it("shows a loading placeholder while overview is pending", () => {
    vi.mocked(apiClient.get).mockReturnValue(new Promise(() => {}));
    render(<Dashboard />);
    expect(screen.getByText("Loading stats…")).toBeInTheDocument();
  });

  it("New Task button invokes navigate to the task list route", async () => {
    render(<Dashboard />);
    fireEvent.click(screen.getByRole("button", { name: /new task/i }));
    expect(mockNavigate).toHaveBeenCalledWith(
      expect.stringContaining("/dashboard/orcamind/tasks"),
    );
  });

  it("Start Experiment button invokes navigate to the experiments route", async () => {
    render(<Dashboard />);
    fireEvent.click(screen.getByRole("button", { name: /start experiment/i }));
    expect(mockNavigate).toHaveBeenCalledWith(
      expect.stringContaining("/dashboard/orcalab/experiments"),
    );
  });

  it("Score Transfer button invokes navigate to the transfer route", async () => {
    render(<Dashboard />);
    fireEvent.click(screen.getByRole("button", { name: /score transfer/i }));
    expect(mockNavigate).toHaveBeenCalledWith(
      expect.stringContaining("/dashboard/orcanet/transfer"),
    );
  });
});
