import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { TaskDetail } from "@/pages/orcamind/TaskDetail";
import apiClient from "@/api/client";
import {
  mockTask,
  mockRecommendations,
  mockSimilarTasks,
  mockPerformancePrediction,
} from "@/test/mocks/handlers";

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

vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>();
  return {
    ...actual,
    useParams: () => ({ id: mockTask.task_id }),
    useNavigate: () => vi.fn(),
  };
});

describe("TaskDetail", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(apiClient.get).mockImplementation((url: string) => {
      if (url === `/orcamind/tasks/${mockTask.task_id}`) {
        return Promise.resolve({ data: mockTask });
      }
      return Promise.reject(new Error(`Unexpected: ${url}`));
    });
    vi.mocked(apiClient.post).mockResolvedValue({ data: [] });
    vi.mocked(apiClient.delete).mockResolvedValue({});
  });

  it("renders the page heading", () => {
    render(<TaskDetail />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "Task Detail",
    );
  });

  it("shows a loading state before task data arrives", () => {
    vi.mocked(apiClient.get).mockReturnValue(new Promise(() => {}));
    render(<TaskDetail />);
    expect(screen.getByTestId("task-loading")).toBeInTheDocument();
  });

  it("renders task metadata once the query resolves", async () => {
    render(<TaskDetail />);
    await waitFor(() => {
      expect(screen.getByTestId("task-metadata")).toBeInTheDocument();
    });
    expect(screen.getByText("Image Classification")).toBeInTheDocument();
    expect(screen.getByText("vision")).toBeInTheDocument();
    expect(screen.getByText("classification")).toBeInTheDocument();
    expect(screen.getByText("50,000")).toBeInTheDocument();
  });

  it("shows feature and class counts when present", async () => {
    render(<TaskDetail />);
    await waitFor(() => {
      expect(screen.getByText("2,048")).toBeInTheDocument();
    });
    expect(screen.getByText("10")).toBeInTheDocument();
  });

  it("shows an error when the task fails to load", async () => {
    vi.mocked(apiClient.get).mockRejectedValue(new Error("not found"));
    render(<TaskDetail />);
    await waitFor(() => {
      expect(screen.getByTestId("task-error")).toBeInTheDocument();
    });
  });

  it("renders the bookmark button", () => {
    render(<TaskDetail />);
    expect(screen.getByTestId("bookmark-btn")).toBeInTheDocument();
  });

  it("toggles bookmark state on click — add bookmark", async () => {
    vi.mocked(apiClient.post).mockImplementation((url: string) => {
      if (url === "/bookmarks") {
        return Promise.resolve({
          data: { id: "bm-1", resource_type: "task", resource_id: mockTask.task_id },
        });
      }
      return Promise.resolve({ data: [] });
    });
    render(<TaskDetail />);
    const btn = screen.getByTestId("bookmark-btn");
    expect(btn).toHaveTextContent("Bookmark");
    fireEvent.click(btn);
    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        "/bookmarks",
        expect.objectContaining({ resource_type: "task" }),
      );
    });
    await waitFor(() => {
      expect(screen.getByTestId("bookmark-btn")).toHaveTextContent("Bookmarked");
    });
  });

  it("renders the Get Recommendations button", () => {
    render(<TaskDetail />);
    expect(screen.getByTestId("get-recommendations-btn")).toBeInTheDocument();
  });

  it("fetches and displays recommendations on button click", async () => {
    vi.mocked(apiClient.post).mockImplementation((url: string) => {
      if (url === "/orcamind/recommend") {
        return Promise.resolve({ data: mockRecommendations });
      }
      return Promise.resolve({ data: [] });
    });
    render(<TaskDetail />);
    fireEvent.click(screen.getByTestId("get-recommendations-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("recommendation-cards")).toBeInTheDocument();
    });
    expect(screen.getByText("ResNet-50")).toBeInTheDocument();
    expect(screen.getByText("EfficientNet-B3")).toBeInTheDocument();
  });

  it("shows an error when recommendations fail to load", async () => {
    vi.mocked(apiClient.post).mockRejectedValue(new Error("server error"));
    render(<TaskDetail />);
    fireEvent.click(screen.getByTestId("get-recommendations-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("rec-error")).toBeInTheDocument();
    });
  });

  it("renders the Find Similar Tasks button", () => {
    render(<TaskDetail />);
    expect(screen.getByTestId("find-similar-btn")).toBeInTheDocument();
  });

  it("fetches and displays similar tasks on button click", async () => {
    vi.mocked(apiClient.post).mockImplementation((url: string) => {
      if (url === "/orcamind/similar-tasks") {
        return Promise.resolve({ data: mockSimilarTasks });
      }
      return Promise.resolve({ data: [] });
    });
    render(<TaskDetail />);
    fireEvent.click(screen.getByTestId("find-similar-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("similar-table")).toBeInTheDocument();
    });
    expect(screen.getByText("Object Detection")).toBeInTheDocument();
    expect(screen.getByText("Face Recognition")).toBeInTheDocument();
  });

  it("renders the model selector and Predict Performance button", () => {
    render(<TaskDetail />);
    expect(screen.getByTestId("model-select")).toBeInTheDocument();
    expect(screen.getByTestId("predict-btn")).toBeInTheDocument();
  });

  it("fetches and displays a performance prediction on button click", async () => {
    vi.mocked(apiClient.post).mockImplementation((url: string) => {
      if (url === "/orcamind/predict-performance") {
        return Promise.resolve({ data: mockPerformancePrediction });
      }
      return Promise.resolve({ data: [] });
    });
    render(<TaskDetail />);
    fireEvent.click(screen.getByTestId("predict-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("prediction-result")).toBeInTheDocument();
    });
    expect(screen.getByText("89.7%")).toBeInTheDocument();
    expect(screen.getByText("76.0%")).toBeInTheDocument();
  });

  it("passes the selected model name to the predict-performance request", async () => {
    vi.mocked(apiClient.post).mockImplementation((url: string) => {
      if (url === "/orcamind/predict-performance") {
        return Promise.resolve({ data: mockPerformancePrediction });
      }
      return Promise.resolve({ data: [] });
    });
    render(<TaskDetail />);
    fireEvent.change(screen.getByTestId("model-select"), {
      target: { value: "XGBoost" },
    });
    fireEvent.click(screen.getByTestId("predict-btn"));
    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        "/orcamind/predict-performance",
        expect.objectContaining({
          model_config: { model_name: "XGBoost" },
        }),
      );
    });
  });
});
