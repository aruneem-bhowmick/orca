import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { ExperimentList } from "@/pages/orcalab/ExperimentList";
import apiClient from "@/api/client";
import type { Experiment } from "@/api/types";

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

const mockNavigate = vi.fn();
vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>();
  return { ...actual, useNavigate: () => mockNavigate };
});

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------

const mockRunning: Experiment = {
  experiment_id: "exp-001",
  name: "Vision Run A",
  task_id: "task-001",
  model_id: "model-001",
  status: "running",
  started_at: "2024-03-15T10:00:00Z",
  completed_at: null,
  training_config: null,
  metrics: null,
  mlflow_run_id: null,
  created_at: "2024-03-15T09:55:00Z",
};

const mockCompleted: Experiment = {
  experiment_id: "exp-002",
  name: "NLP Run B",
  task_id: "task-002",
  model_id: "model-002",
  status: "completed",
  started_at: "2024-03-14T08:00:00Z",
  completed_at: "2024-03-14T10:00:00Z",
  training_config: null,
  metrics: { accuracy: 0.95, loss: 0.1 },
  mlflow_run_id: "mlf-run-001",
  created_at: "2024-03-14T07:55:00Z",
};

const mockFailed: Experiment = {
  experiment_id: "exp-003",
  name: "Failed Run C",
  task_id: "task-003",
  model_id: "model-003",
  status: "failed",
  started_at: "2024-03-13T06:00:00Z",
  completed_at: "2024-03-13T06:30:00Z",
  training_config: null,
  metrics: null,
  mlflow_run_id: null,
  created_at: "2024-03-13T05:55:00Z",
};

const mockExperiments: Experiment[] = [mockRunning, mockCompleted, mockFailed];

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(apiClient.get).mockImplementation((url: string) => {
    if (url === "/orcalab/experiments") {
      return Promise.resolve({ data: mockExperiments });
    }
    return Promise.reject(new Error(`Unexpected: ${url}`));
  });
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ExperimentList", () => {
  it("renders the page heading", () => {
    render(<ExperimentList />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "Experiments",
    );
  });

  it("renders the New Experiment button", () => {
    render(<ExperimentList />);
    expect(screen.getByTestId("new-experiment-btn")).toBeInTheDocument();
  });

  it("shows a skeleton loading state before data arrives", () => {
    vi.mocked(apiClient.get).mockReturnValue(new Promise(() => {}));
    render(<ExperimentList />);
    expect(screen.getByTestId("table-skeleton")).toBeInTheDocument();
  });

  it("renders the experiment table after data loads", async () => {
    render(<ExperimentList />);
    await waitFor(() => {
      expect(screen.getByTestId("exp-table")).toBeInTheDocument();
    });
  });

  it("renders a row for each experiment", async () => {
    render(<ExperimentList />);
    await waitFor(() => {
      expect(screen.getByTestId("exp-row-exp-001")).toBeInTheDocument();
      expect(screen.getByTestId("exp-row-exp-002")).toBeInTheDocument();
      expect(screen.getByTestId("exp-row-exp-003")).toBeInTheDocument();
    });
  });

  it("shows the experiment name in each row", async () => {
    render(<ExperimentList />);
    await waitFor(() => {
      expect(screen.getByText("Vision Run A")).toBeInTheDocument();
      expect(screen.getByText("NLP Run B")).toBeInTheDocument();
      expect(screen.getByText("Failed Run C")).toBeInTheDocument();
    });
  });

  it("renders a pulsing indicator for running experiments", async () => {
    render(<ExperimentList />);
    await waitFor(() => {
      const badge = screen.getByTestId("status-badge-running");
      expect(badge.querySelector(".animate-pulse")).toBeInTheDocument();
    });
  });

  it("shows status badges for each experiment", async () => {
    render(<ExperimentList />);
    await waitFor(() => {
      expect(screen.getByTestId("status-badge-running")).toBeInTheDocument();
      expect(screen.getByTestId("status-badge-completed")).toBeInTheDocument();
      expect(screen.getByTestId("status-badge-failed")).toBeInTheDocument();
    });
  });

  it("shows an empty state when no experiments are returned", async () => {
    vi.mocked(apiClient.get).mockResolvedValue({ data: [] });
    render(<ExperimentList />);
    await waitFor(() => {
      expect(screen.getByTestId("exp-list-empty")).toBeInTheDocument();
    });
  });

  it("shows an error state when the API call fails", async () => {
    vi.mocked(apiClient.get).mockRejectedValue(new Error("network error"));
    render(<ExperimentList />);
    await waitFor(() => {
      expect(screen.getByTestId("exp-list-error")).toBeInTheDocument();
    });
  });

  it("navigates to experiment detail when a row is clicked", async () => {
    render(<ExperimentList />);
    await waitFor(() => {
      expect(screen.getByTestId("exp-row-exp-001")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("exp-row-exp-001"));
    expect(mockNavigate).toHaveBeenCalledWith(expect.stringContaining("exp-001"));
  });

  it("navigates when Enter is pressed on a row", async () => {
    render(<ExperimentList />);
    await waitFor(() => {
      expect(screen.getByTestId("exp-row-exp-001")).toBeInTheDocument();
    });
    fireEvent.keyDown(screen.getByTestId("exp-row-exp-001"), { key: "Enter" });
    expect(mockNavigate).toHaveBeenCalledWith(expect.stringContaining("exp-001"));
  });

  it("filters to only running experiments when status filter is set to 'running'", async () => {
    render(<ExperimentList />);
    await waitFor(() => {
      expect(screen.getByTestId("exp-table")).toBeInTheDocument();
    });
    fireEvent.change(screen.getByTestId("status-filter"), {
      target: { value: "running" },
    });
    await waitFor(() => {
      expect(screen.getByText("Vision Run A")).toBeInTheDocument();
      expect(screen.queryByText("NLP Run B")).not.toBeInTheDocument();
      expect(screen.queryByText("Failed Run C")).not.toBeInTheDocument();
    });
  });

  it("filters to only completed experiments when status filter is set to 'completed'", async () => {
    render(<ExperimentList />);
    await waitFor(() => {
      expect(screen.getByTestId("exp-table")).toBeInTheDocument();
    });
    fireEvent.change(screen.getByTestId("status-filter"), {
      target: { value: "completed" },
    });
    await waitFor(() => {
      expect(screen.queryByText("Vision Run A")).not.toBeInTheDocument();
      expect(screen.getByText("NLP Run B")).toBeInTheDocument();
    });
  });

  it("shows empty state when filter produces no matches", async () => {
    vi.mocked(apiClient.get).mockResolvedValue({ data: [mockRunning] });
    render(<ExperimentList />);
    await waitFor(() => {
      expect(screen.getByTestId("exp-table")).toBeInTheDocument();
    });
    fireEvent.change(screen.getByTestId("status-filter"), {
      target: { value: "completed" },
    });
    await waitFor(() => {
      expect(screen.getByTestId("exp-list-empty")).toBeInTheDocument();
    });
  });

  it("opens the new experiment dialog when the button is clicked", async () => {
    render(<ExperimentList />);
    fireEvent.click(screen.getByTestId("new-experiment-btn"));
    expect(screen.getByTestId("new-experiment-dialog")).toBeInTheDocument();
  });

  it("closes the new experiment dialog when Cancel is clicked", () => {
    render(<ExperimentList />);
    fireEvent.click(screen.getByTestId("new-experiment-btn"));
    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(screen.queryByTestId("new-experiment-dialog")).not.toBeInTheDocument();
  });

  it("shows validation errors when dialog is submitted empty", () => {
    render(<ExperimentList />);
    fireEvent.click(screen.getByTestId("new-experiment-btn"));
    fireEvent.click(screen.getByTestId("exp-submit-btn"));
    expect(screen.getByText("Task ID is required")).toBeInTheDocument();
    expect(screen.getByText("Model ID is required")).toBeInTheDocument();
  });

  it("calls POST /orcalab/experiments and closes dialog on valid submit", async () => {
    vi.mocked(apiClient.post).mockResolvedValue({ data: mockRunning });
    render(<ExperimentList />);
    fireEvent.click(screen.getByTestId("new-experiment-btn"));

    fireEvent.change(screen.getByTestId("exp-task-id"), {
      target: { value: "task-001" },
    });
    fireEvent.change(screen.getByTestId("exp-model-id"), {
      target: { value: "model-001" },
    });
    fireEvent.click(screen.getByTestId("exp-submit-btn"));

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        "/orcalab/experiments",
        expect.objectContaining({
          task_id: "task-001",
          model_id: "model-001",
        }),
      );
    });
    await waitFor(() => {
      expect(screen.queryByTestId("new-experiment-dialog")).not.toBeInTheDocument();
    });
  });

  it("pre-fills task_id and model_id from query params in the new experiment dialog", async () => {
    const { MemoryRouter } = await import("react-router-dom");
    const { QueryClient, QueryClientProvider } = await import("@tanstack/react-query");
    const { render: tlRender } = await import("@testing-library/react");

    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false, gcTime: 0 } }
    });

    tlRender(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={["/orcalab/experiments?task_id=task-abc&model_id=model-xyz"]}>
          <ExperimentList />
        </MemoryRouter>
      </QueryClientProvider>
    );

    fireEvent.click(screen.getByTestId("new-experiment-btn"));

    expect(screen.getByTestId("new-experiment-dialog")).toBeInTheDocument();
    expect(screen.getByTestId("exp-task-id")).toHaveValue("task-abc");
    expect(screen.getByTestId("exp-model-id")).toHaveValue("model-xyz");
  });
});
