import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { SweepManager } from "@/pages/orcalab/SweepManager";
import apiClient from "@/api/client";
import type { Sweep, Task } from "@/api/types";

// ---------------------------------------------------------------------------
// Mock apiClient
// ---------------------------------------------------------------------------

vi.mock("@/api/client", () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    interceptors: {
      request: { use: vi.fn(), handlers: [] },
      response: { use: vi.fn(), handlers: [] },
    },
  },
}));

// ---------------------------------------------------------------------------
// Mock recharts to avoid SVG rendering issues in jsdom
// ---------------------------------------------------------------------------

vi.mock("recharts", () => ({
  LineChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="line-chart">{children}</div>
  ),
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
}));

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------

const mockTask: Task = {
  task_id: "task-001",
  name: "Image Classification",
  domain: "vision",
  task_type: "classification",
  n_samples: 50000,
  n_features: 2048,
  n_classes: 10,
  metadata: null,
  created_at: "2024-03-15T10:00:00Z",
};

const mockSweepRunning: Sweep = {
  sweep_id: "sweep-001",
  task_id: "task-001",
  search_strategy: "random",
  n_trials: 50,
  completed_trials: 23,
  status: "running",
  best_trial: null,
  results: null,
  created_at: "2024-03-15T10:00:00Z",
};

const mockSweepCompleted: Sweep = {
  sweep_id: "sweep-002",
  task_id: "task-002",
  search_strategy: "bayesian",
  n_trials: 30,
  completed_trials: 30,
  status: "completed",
  best_trial: 3,
  results: [
    {
      trial_id: 1,
      params: { lr: 0.001, batch_size: 32 },
      metrics: { accuracy: 0.88, loss: 0.32 },
    },
    {
      trial_id: 2,
      params: { lr: 0.01, batch_size: 64 },
      metrics: { accuracy: 0.91, loss: 0.25 },
    },
    {
      trial_id: 3,
      params: { lr: 0.005, batch_size: 32 },
      metrics: { accuracy: 0.94, loss: 0.19 },
    },
  ],
  created_at: "2024-03-14T08:00:00Z",
};

const mockSweeps: Sweep[] = [mockSweepRunning, mockSweepCompleted];

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(apiClient.get).mockImplementation((url: string) => {
    if (url === "/orcalab/sweeps") {
      return Promise.resolve({ data: mockSweeps });
    }
    if (url === "/orcamind/tasks") {
      return Promise.resolve({ data: [mockTask] });
    }
    return Promise.reject(new Error(`Unexpected: ${url}`));
  });
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("SweepManager", () => {
  it("renders the page heading", () => {
    render(<SweepManager />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Sweeps");
  });

  it("renders the New Sweep button", () => {
    render(<SweepManager />);
    expect(screen.getByTestId("new-sweep-btn")).toBeInTheDocument();
  });

  it("shows a loading state before data arrives", () => {
    vi.mocked(apiClient.get).mockReturnValue(new Promise(() => {}));
    render(<SweepManager />);
    expect(screen.getByTestId("sweep-list-loading")).toBeInTheDocument();
  });

  it("renders a row for each sweep", async () => {
    render(<SweepManager />);
    await waitFor(() => {
      expect(screen.getByTestId("sweep-row-sweep-001")).toBeInTheDocument();
      expect(screen.getByTestId("sweep-row-sweep-002")).toBeInTheDocument();
    });
  });

  it("shows sweep ID, strategy, and trial counts in rows", async () => {
    render(<SweepManager />);
    await waitFor(() => {
      expect(screen.getByText("sweep-001")).toBeInTheDocument();
      expect(screen.getByText("random")).toBeInTheDocument();
    });
  });

  it("shows empty state when no sweeps exist", async () => {
    vi.mocked(apiClient.get).mockImplementation((url: string) => {
      if (url === "/orcalab/sweeps") return Promise.resolve({ data: [] });
      if (url === "/orcamind/tasks") return Promise.resolve({ data: [mockTask] });
      return Promise.reject(new Error(`Unexpected: ${url}`));
    });
    render(<SweepManager />);
    await waitFor(() => {
      expect(screen.getByTestId("sweep-list-empty")).toBeInTheDocument();
    });
  });

  it("shows error state when fetch fails", async () => {
    vi.mocked(apiClient.get).mockRejectedValue(new Error("network error"));
    render(<SweepManager />);
    await waitFor(() => {
      expect(screen.getByTestId("sweep-list-error")).toBeInTheDocument();
    });
  });

  it("expands sweep detail panel when a row is clicked", async () => {
    render(<SweepManager />);
    await waitFor(() => {
      expect(screen.getByTestId("sweep-row-sweep-001")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("sweep-row-sweep-001"));
    await waitFor(() => {
      expect(screen.getByTestId("sweep-detail-sweep-001")).toBeInTheDocument();
    });
  });

  it("collapses sweep detail panel when the same row is clicked again", async () => {
    render(<SweepManager />);
    await waitFor(() => {
      expect(screen.getByTestId("sweep-row-sweep-001")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("sweep-row-sweep-001"));
    await waitFor(() => {
      expect(screen.getByTestId("sweep-detail-sweep-001")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("sweep-row-sweep-001"));
    await waitFor(() => {
      expect(screen.queryByTestId("sweep-detail-sweep-001")).not.toBeInTheDocument();
    });
  });

  it("expands row via keyboard Enter", async () => {
    render(<SweepManager />);
    await waitFor(() => {
      expect(screen.getByTestId("sweep-row-sweep-001")).toBeInTheDocument();
    });
    fireEvent.keyDown(screen.getByTestId("sweep-row-sweep-001"), { key: "Enter" });
    await waitFor(() => {
      expect(screen.getByTestId("sweep-detail-sweep-001")).toBeInTheDocument();
    });
  });

  it("shows trial progress bar in expanded detail panel", async () => {
    render(<SweepManager />);
    await waitFor(() => {
      expect(screen.getByTestId("sweep-row-sweep-001")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("sweep-row-sweep-001"));
    await waitFor(() => {
      expect(screen.getByTestId("trial-progress-bar")).toBeInTheDocument();
    });
  });

  it("shows 'No completed trials yet' for a sweep with no results", async () => {
    render(<SweepManager />);
    await waitFor(() => {
      expect(screen.getByTestId("sweep-row-sweep-001")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("sweep-row-sweep-001"));
    await waitFor(() => {
      expect(screen.getByTestId("no-trials-yet")).toBeInTheDocument();
    });
  });

  it("renders the sweep results chart and trials table when results exist", async () => {
    render(<SweepManager />);
    await waitFor(() => {
      expect(screen.getByTestId("sweep-row-sweep-002")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("sweep-row-sweep-002"));
    await waitFor(() => {
      expect(screen.getByTestId("sweep-chart")).toBeInTheDocument();
      expect(screen.getByTestId("sweep-trials-table")).toBeInTheDocument();
    });
  });

  it("highlights the best trial row", async () => {
    render(<SweepManager />);
    await waitFor(() => {
      expect(screen.getByTestId("sweep-row-sweep-002")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("sweep-row-sweep-002"));
    await waitFor(() => {
      const bestRow = screen.getByTestId("trial-row-3");
      expect(bestRow.className).toMatch(/amber/);
    });
  });

  it("shows the best-trial star marker", async () => {
    render(<SweepManager />);
    await waitFor(() => {
      expect(screen.getByTestId("sweep-row-sweep-002")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("sweep-row-sweep-002"));
    await waitFor(() => {
      const bestRow = screen.getByTestId("trial-row-3");
      expect(bestRow.textContent).toContain("★");
    });
  });

  it("opens the new sweep dialog when the button is clicked", async () => {
    render(<SweepManager />);
    await waitFor(() => {
      expect(screen.getByTestId("new-sweep-btn")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("new-sweep-btn"));
    expect(screen.getByTestId("new-sweep-dialog")).toBeInTheDocument();
  });

  it("closes the dialog when Cancel is clicked", async () => {
    render(<SweepManager />);
    await waitFor(() => {
      expect(screen.getByTestId("new-sweep-btn")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("new-sweep-btn"));
    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(screen.queryByTestId("new-sweep-dialog")).not.toBeInTheDocument();
  });

  it("populates the task dropdown with available tasks", async () => {
    render(<SweepManager />);
    await waitFor(() => {
      expect(screen.getByTestId("new-sweep-btn")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("new-sweep-btn"));
    await waitFor(() => {
      const select = screen.getByTestId("sweep-task-select");
      expect(select.textContent).toContain("Image Classification");
    });
  });

  it("calls POST /orcalab/sweeps and closes dialog on valid submit", async () => {
    vi.mocked(apiClient.post).mockResolvedValue({ data: mockSweepRunning });
    render(<SweepManager />);
    await waitFor(() => {
      expect(screen.getByTestId("new-sweep-btn")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("new-sweep-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("sweep-task-select")).toBeInTheDocument();
    });

    fireEvent.change(screen.getByTestId("sweep-n-trials"), {
      target: { value: "10" },
    });
    fireEvent.click(screen.getByTestId("sweep-submit-btn"));

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        "/orcalab/sweeps",
        {
          task_id: "task-001",
          search_strategy: "random",
          use_orcamind_priors: false,
          n_trials: 10,
        },
      );
    });
    await waitFor(() => {
      expect(screen.queryByTestId("new-sweep-dialog")).not.toBeInTheDocument();
    });
  });

  it("handles tasks loading asynchronously after dialog is opened", async () => {
    let resolveTasks: (value: any) => void = () => {};
    const tasksPromise = new Promise((resolve) => {
      resolveTasks = resolve;
    });

    vi.mocked(apiClient.get).mockImplementation((url: string) => {
      if (url === "/orcalab/sweeps") {
        return Promise.resolve({ data: mockSweeps });
      }
      if (url === "/orcamind/tasks") {
        return tasksPromise;
      }
      return Promise.reject(new Error(`Unexpected: ${url}`));
    });

    render(<SweepManager />);
    await waitFor(() => {
      expect(screen.getByTestId("new-sweep-btn")).toBeInTheDocument();
    });

    // Open dialog immediately while tasks are still loading
    fireEvent.click(screen.getByTestId("new-sweep-btn"));
    expect(screen.getByTestId("new-sweep-dialog")).toBeInTheDocument();

    // Verify task selection is initially empty
    const select = screen.getByTestId("sweep-task-select");
    expect(select).toHaveValue("");

    // Now resolve the tasks request
    const { act } = await import("@testing-library/react");
    await act(async () => {
      resolveTasks({ data: [mockTask] });
    });

    // Verify dropdown is populated and taskId is updated
    await waitFor(() => {
      expect(select).toHaveValue(mockTask.task_id);
    });
    expect(select.textContent).toContain("Image Classification");

    // Submit and check it does not submit an empty taskId
    vi.mocked(apiClient.post).mockResolvedValue({ data: mockSweepRunning });
    fireEvent.click(screen.getByTestId("sweep-submit-btn"));

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        "/orcalab/sweeps",
        {
          task_id: "task-001",
          search_strategy: "random",
          use_orcamind_priors: false,
          n_trials: 20,
        },
      );
    });
  });

  it("toggles the OrcaMind priors checkbox", async () => {
    render(<SweepManager />);
    await waitFor(() => {
      expect(screen.getByTestId("new-sweep-btn")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("new-sweep-btn"));
    const checkbox = screen.getByTestId("sweep-use-priors") as HTMLInputElement;
    expect(checkbox.checked).toBe(false);
    fireEvent.click(checkbox);
    expect(checkbox.checked).toBe(true);
  });

  it("rejects fractional trial counts", async () => {
    render(<SweepManager />);
    await waitFor(() => {
      expect(screen.getByTestId("new-sweep-btn")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("new-sweep-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("sweep-task-select")).toBeInTheDocument();
    });

    fireEvent.change(screen.getByTestId("sweep-n-trials"), {
      target: { value: "1.5" },
    });
    fireEvent.click(screen.getByTestId("sweep-submit-btn"));

    await waitFor(() => {
      expect(screen.getByText("Number of trials must be a whole number")).toBeInTheDocument();
    });
  });
});
