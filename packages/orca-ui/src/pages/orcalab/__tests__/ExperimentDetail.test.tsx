import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { ExperimentDetail } from "@/pages/orcalab/ExperimentDetail";
import apiClient from "@/api/client";
import type { Experiment, MetricUpdate } from "@/api/types";

// ---------------------------------------------------------------------------
// Mock apiClient
// ---------------------------------------------------------------------------

vi.mock("@/api/client", () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    delete: vi.fn(),
    interceptors: {
      request: { use: vi.fn(), handlers: [] },
      response: { use: vi.fn(), handlers: [] },
    },
  },
}));

// ---------------------------------------------------------------------------
// Mock react-router-dom so useParams returns a fixed id
// ---------------------------------------------------------------------------

vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>();
  return { ...actual, useParams: () => ({ id: "exp-001" }) };
});

// ---------------------------------------------------------------------------
// Mock useWebSocket so we can control messages and connection state
// ---------------------------------------------------------------------------

type WsControls = {
  setIsConnected: (v: boolean) => void;
  pushMessage: (m: MetricUpdate) => void;
  sendSpy: ReturnType<typeof vi.fn>;
  closeSpy: ReturnType<typeof vi.fn>;
};

let wsControls: WsControls;

vi.mock("@/hooks/useWebSocket", () => ({
  useWebSocket: () => {
    const { useState } = require("react");
    const [messages, setMessages] = useState<MetricUpdate[]>([]);
    const [isConnected, setIsConnected] = useState(false);
    const sendSpy = vi.fn();
    const closeSpy = vi.fn();

    wsControls = {
      setIsConnected,
      pushMessage: (m: MetricUpdate) => setMessages((prev: MetricUpdate[]) => [...prev, m]),
      sendSpy,
      closeSpy,
    };

    return { messages, isConnected, send: sendSpy, close: closeSpy };
  },
}));

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------

const runningExperiment: Experiment = {
  experiment_id: "exp-001",
  name: "Running Vision Experiment",
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

const completedExperiment: Experiment = {
  experiment_id: "exp-001",
  name: "Completed NLP Experiment",
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

const failedExperiment: Experiment = {
  ...completedExperiment,
  name: "Failed Experiment",
  status: "failed",
  metrics: null,
  mlflow_run_id: null,
};

const pendingExperiment: Experiment = {
  ...completedExperiment,
  name: "Pending Experiment",
  status: "pending",
  metrics: null,
  started_at: null,
  completed_at: null,
  mlflow_run_id: null,
};

const mockMetricUpdate: MetricUpdate = {
  epoch: 1,
  loss: 0.5,
  accuracy: 0.75,
  learning_rate: 0.001,
  timestamp: "2024-03-15T10:01:00Z",
};

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(apiClient.get).mockResolvedValue({ data: runningExperiment });
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ExperimentDetail", () => {
  it("renders the page heading", () => {
    render(<ExperimentDetail />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "Experiment Detail",
    );
  });

  it("shows a loading state before data arrives", () => {
    vi.mocked(apiClient.get).mockReturnValue(new Promise(() => {}));
    render(<ExperimentDetail />);
    expect(screen.getByTestId("exp-loading")).toBeInTheDocument();
  });

  it("shows an error state when the API call fails", async () => {
    vi.mocked(apiClient.get).mockRejectedValue(new Error("network error"));
    render(<ExperimentDetail />);
    await waitFor(() => {
      expect(screen.getByTestId("exp-error")).toBeInTheDocument();
    });
  });

  it("renders experiment metadata in a definition list", async () => {
    render(<ExperimentDetail />);
    await waitFor(() => {
      expect(screen.getByTestId("exp-metadata")).toBeInTheDocument();
    });
    expect(screen.getByText("Running Vision Experiment")).toBeInTheDocument();
    expect(screen.getByText("running")).toBeInTheDocument();
  });

  it("renders the bookmark button", () => {
    render(<ExperimentDetail />);
    expect(screen.getByTestId("bookmark-btn")).toBeInTheDocument();
  });

  it("hydrates bookmarked state from existing bookmarks on load", async () => {
    vi.mocked(apiClient.get).mockImplementation((url: string) => {
      if (url.includes("/bookmarks")) {
        return Promise.resolve({
          data: {
            items: [
              { id: "bm-001", resource_type: "experiment", resource_id: "exp-001" }
            ]
          }
        });
      }
      return Promise.resolve({ data: runningExperiment });
    });

    render(<ExperimentDetail />);
    
    await waitFor(() => {
      expect(screen.getByTestId("bookmark-btn")).toHaveTextContent("Bookmarked");
    });

    vi.mocked(apiClient.delete).mockResolvedValue({});
    fireEvent.click(screen.getByTestId("bookmark-btn"));
    await waitFor(() => {
      expect(apiClient.delete).toHaveBeenCalledWith("/bookmarks/bm-001");
    });
  });

  it("toggles bookmark on when clicked (not bookmarked -> bookmarked)", async () => {
    vi.mocked(apiClient.post).mockResolvedValue({
      data: { id: "bm-001", resource_type: "experiment", resource_id: "exp-001" },
    });
    render(<ExperimentDetail />);
    const btn = screen.getByTestId("bookmark-btn");
    fireEvent.click(btn);
    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        "/bookmarks",
        expect.objectContaining({ resource_type: "experiment", resource_id: "exp-001" }),
      );
    });
    await waitFor(() => {
      expect(screen.getByTestId("bookmark-btn")).toHaveTextContent("Bookmarked");
    });
  });

  it("removes bookmark when clicked while bookmarked", async () => {
    vi.mocked(apiClient.post).mockResolvedValue({
      data: { id: "bm-001", resource_type: "experiment", resource_id: "exp-001" },
    });
    vi.mocked(apiClient.delete).mockResolvedValue({});
    render(<ExperimentDetail />);
    fireEvent.click(screen.getByTestId("bookmark-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("bookmark-btn")).toHaveTextContent("Bookmarked");
    });
    fireEvent.click(screen.getByTestId("bookmark-btn"));
    await waitFor(() => {
      expect(apiClient.delete).toHaveBeenCalledWith("/bookmarks/bm-001");
    });
  });

  // -- Live metrics section (status = running) --------------------------------

  it("renders the live metrics section for a running experiment", async () => {
    render(<ExperimentDetail />);
    await waitFor(() => {
      expect(screen.getByTestId("live-metrics")).toBeInTheDocument();
    });
  });

  it("shows 'Waiting for metric data…' when no messages have arrived yet", async () => {
    render(<ExperimentDetail />);
    await waitFor(() => {
      expect(screen.getByTestId("no-metrics-yet")).toBeInTheDocument();
    });
  });

  it("shows the ws-status indicator as disconnected initially", async () => {
    render(<ExperimentDetail />);
    await waitFor(() => {
      expect(screen.getByTestId("ws-status")).toHaveTextContent("Reconnecting");
    });
  });

  it("shows ws-status as Live when connected", async () => {
    render(<ExperimentDetail />);
    await waitFor(() => {
      expect(screen.getByTestId("live-metrics")).toBeInTheDocument();
    });
    vi.mocked(apiClient.get); // just to satisfy the await pattern
    // Simulate connection becoming live
    const { act } = await import("@testing-library/react");
    await act(async () => {
      wsControls.setIsConnected(true);
    });
    await waitFor(() => {
      expect(screen.getByTestId("ws-status")).toHaveTextContent("Live");
    });
  });

  it("renders the metric chart when messages arrive", async () => {
    render(<ExperimentDetail />);
    await waitFor(() => {
      expect(screen.getByTestId("live-metrics")).toBeInTheDocument();
    });
    const { act } = await import("@testing-library/react");
    await act(async () => {
      wsControls.pushMessage(mockMetricUpdate);
    });
    await waitFor(() => {
      expect(screen.getByTestId("metric-chart")).toBeInTheDocument();
      expect(screen.queryByTestId("no-metrics-yet")).not.toBeInTheDocument();
    });
  });

  it("renders the metric table when messages arrive", async () => {
    render(<ExperimentDetail />);
    await waitFor(() => {
      expect(screen.getByTestId("live-metrics")).toBeInTheDocument();
    });
    const { act } = await import("@testing-library/react");
    await act(async () => {
      wsControls.pushMessage(mockMetricUpdate);
    });
    await waitFor(() => {
      expect(screen.getByTestId("metric-table")).toBeInTheDocument();
    });
  });

  it("sends a pause control message when Pause is clicked", async () => {
    render(<ExperimentDetail />);
    await waitFor(() => {
      expect(screen.getByTestId("live-metrics")).toBeInTheDocument();
    });
    const { act } = await import("@testing-library/react");
    await act(async () => {
      wsControls.setIsConnected(true);
    });
    fireEvent.click(screen.getByTestId("pause-btn"));
    expect(wsControls.sendSpy).toHaveBeenCalledWith({ action: "pause" });
  });

  it("sends a resume control message when Resume is clicked", async () => {
    render(<ExperimentDetail />);
    await waitFor(() => {
      expect(screen.getByTestId("live-metrics")).toBeInTheDocument();
    });
    const { act } = await import("@testing-library/react");
    await act(async () => {
      wsControls.setIsConnected(true);
    });
    fireEvent.click(screen.getByTestId("resume-btn"));
    expect(wsControls.sendSpy).toHaveBeenCalledWith({ action: "resume" });
  });

  it("sends a cancel control message and calls close() when Cancel is clicked", async () => {
    render(<ExperimentDetail />);
    await waitFor(() => {
      expect(screen.getByTestId("live-metrics")).toBeInTheDocument();
    });
    const { act } = await import("@testing-library/react");
    await act(async () => {
      wsControls.setIsConnected(true);
    });
    fireEvent.click(screen.getByTestId("cancel-btn"));
    expect(wsControls.sendSpy).toHaveBeenCalledWith({ action: "cancel" });
    expect(wsControls.closeSpy).toHaveBeenCalled();
  });

  // -- Completed metrics section (status = completed) -----------------------

  it("renders the completed metrics section for a completed experiment", async () => {
    vi.mocked(apiClient.get).mockResolvedValue({ data: completedExperiment });
    render(<ExperimentDetail />);
    await waitFor(() => {
      expect(screen.getByTestId("completed-metrics")).toBeInTheDocument();
    });
  });

  it("shows final metric cards for each metric in the completed experiment", async () => {
    vi.mocked(apiClient.get).mockResolvedValue({ data: completedExperiment });
    render(<ExperimentDetail />);
    await waitFor(() => {
      expect(screen.getByTestId("final-metrics-grid")).toBeInTheDocument();
    });
    expect(screen.getByText("accuracy")).toBeInTheDocument();
    expect(screen.getByText("loss")).toBeInTheDocument();
  });

  it("shows the artifact download link when mlflow_run_id is present", async () => {
    vi.mocked(apiClient.get).mockResolvedValue({ data: completedExperiment });
    render(<ExperimentDetail />);
    await waitFor(() => {
      expect(screen.getByTestId("artifact-download")).toBeInTheDocument();
    });
  });

  it("does not show artifact section when mlflow_run_id is absent", async () => {
    vi.mocked(apiClient.get).mockResolvedValue({
      data: { ...completedExperiment, mlflow_run_id: null },
    });
    render(<ExperimentDetail />);
    await waitFor(() => {
      expect(screen.getByTestId("completed-metrics")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("artifact-section")).not.toBeInTheDocument();
  });

  // -- Other statuses -------------------------------------------------------

  it("shows a failure message for failed experiments", async () => {
    vi.mocked(apiClient.get).mockResolvedValue({ data: failedExperiment });
    render(<ExperimentDetail />);
    await waitFor(() => {
      expect(screen.getByTestId("no-metrics-msg")).toHaveTextContent(
        "Experiment failed",
      );
    });
  });

  it("shows a 'not started yet' message for pending experiments", async () => {
    vi.mocked(apiClient.get).mockResolvedValue({ data: pendingExperiment });
    render(<ExperimentDetail />);
    await waitFor(() => {
      expect(screen.getByTestId("no-metrics-msg")).toHaveTextContent(
        "not started",
      );
    });
  });
});
