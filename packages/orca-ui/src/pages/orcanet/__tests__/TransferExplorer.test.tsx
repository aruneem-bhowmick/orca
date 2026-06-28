import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { TransferExplorer } from "@/pages/orcanet/TransferExplorer";
import apiClient from "@/api/client";
import {
  mockTaskList,
  mockTransferScore,
  mockTransferRecommendations,
  mockExplainResult,
} from "@/test/mocks/handlers";

vi.mock("@/api/client", () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    patch: vi.fn(),
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
// Setup
// ---------------------------------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(apiClient.get).mockImplementation((url: string) => {
    if (url === "/orcamind/tasks") return Promise.resolve({ data: mockTaskList });
    return Promise.reject(new Error(`Unexpected GET: ${url}`));
  });
  vi.mocked(apiClient.post).mockImplementation((url: string) => {
    if (url === "/orcanet/transfer/score") return Promise.resolve({ data: mockTransferScore });
    if (url === "/orcanet/transfer/recommend") return Promise.resolve({ data: mockTransferRecommendations });
    if (url === "/orcanet/explain") return Promise.resolve({ data: mockExplainResult });
    return Promise.reject(new Error(`Unexpected POST: ${url}`));
  });
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("TransferExplorer", () => {
  it("renders the page heading", () => {
    render(<TransferExplorer />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Transfer Explorer");
  });

  it("renders source and target task selects", () => {
    render(<TransferExplorer />);
    expect(screen.getByTestId("source-task-select")).toBeInTheDocument();
    expect(screen.getByTestId("target-task-select")).toBeInTheDocument();
  });

  it("populates task dropdowns from GET /orcamind/tasks", async () => {
    render(<TransferExplorer />);
    await waitFor(() => {
      // Both selects should have task options
      const selects = screen.getAllByRole("combobox");
      expect(selects.length).toBeGreaterThanOrEqual(2);
    });
  });

  it("disables score button when tasks are not selected", () => {
    render(<TransferExplorer />);
    expect(screen.getByTestId("score-btn")).toBeDisabled();
  });

  it("enables score button when different source and target tasks are selected", async () => {
    render(<TransferExplorer />);
    await waitFor(() => {
      expect(screen.getByTestId("source-task-select")).not.toBeDisabled();
    });
    fireEvent.change(screen.getByTestId("source-task-select"), {
      target: { value: mockTaskList[0].task_id },
    });
    fireEvent.change(screen.getByTestId("target-task-select"), {
      target: { value: mockTaskList[1].task_id },
    });
    expect(screen.getByTestId("score-btn")).not.toBeDisabled();
  });

  it("calls POST /orcanet/transfer/score and renders score gauge", async () => {
    render(<TransferExplorer />);
    await waitFor(() => expect(screen.getByTestId("source-task-select")).not.toBeDisabled());

    fireEvent.change(screen.getByTestId("source-task-select"), {
      target: { value: mockTaskList[0].task_id },
    });
    fireEvent.change(screen.getByTestId("target-task-select"), {
      target: { value: mockTaskList[1].task_id },
    });
    fireEvent.click(screen.getByTestId("score-btn"));

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        "/orcanet/transfer/score",
        expect.objectContaining({
          source_task_id: mockTaskList[0].task_id,
          target_task_id: mockTaskList[1].task_id,
        }),
      );
    });
    await waitFor(() => {
      expect(screen.getByTestId("score-gauge")).toBeInTheDocument();
    });
  });

  it("displays the numeric score percentage in the gauge", async () => {
    render(<TransferExplorer />);
    await waitFor(() => expect(screen.getByTestId("source-task-select")).not.toBeDisabled());

    fireEvent.change(screen.getByTestId("source-task-select"), {
      target: { value: mockTaskList[0].task_id },
    });
    fireEvent.change(screen.getByTestId("target-task-select"), {
      target: { value: mockTaskList[1].task_id },
    });
    fireEvent.click(screen.getByTestId("score-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("score-value")).toHaveTextContent("82%");
    });
  });

  it("shows score error when POST /orcanet/transfer/score fails", async () => {
    vi.mocked(apiClient.post).mockRejectedValue(new Error("network"));
    render(<TransferExplorer />);
    await waitFor(() => expect(screen.getByTestId("source-task-select")).not.toBeDisabled());

    fireEvent.change(screen.getByTestId("source-task-select"), {
      target: { value: mockTaskList[0].task_id },
    });
    fireEvent.change(screen.getByTestId("target-task-select"), {
      target: { value: mockTaskList[1].task_id },
    });
    fireEvent.click(screen.getByTestId("score-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("score-error")).toBeInTheDocument();
    });
  });

  it("calls POST /orcanet/transfer/recommend and renders recommendation cards", async () => {
    render(<TransferExplorer />);
    await waitFor(() => expect(screen.getByTestId("target-task-select")).not.toBeDisabled());

    fireEvent.change(screen.getByTestId("target-task-select"), {
      target: { value: mockTaskList[1].task_id },
    });
    fireEvent.click(screen.getByTestId("recommend-btn"));

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        "/orcanet/transfer/recommend",
        expect.objectContaining({ target_task_id: mockTaskList[1].task_id }),
      );
    });
    await waitFor(() => {
      expect(screen.getByTestId("recommendations-list")).toBeInTheDocument();
      expect(
        screen.getByTestId(`rec-card-${mockTransferRecommendations[0].source_task_id}`),
      ).toBeInTheDocument();
    });
  });

  it("shows recommendation error when POST /orcanet/transfer/recommend fails", async () => {
    vi.mocked(apiClient.post).mockRejectedValue(new Error("network"));
    render(<TransferExplorer />);
    await waitFor(() => expect(screen.getByTestId("target-task-select")).not.toBeDisabled());

    fireEvent.change(screen.getByTestId("target-task-select"), {
      target: { value: mockTaskList[1].task_id },
    });
    fireEvent.click(screen.getByTestId("recommend-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("recommend-error")).toBeInTheDocument();
    });
  });

  it("calls POST /orcanet/explain when Explain is clicked and shows explanation panel", async () => {
    render(<TransferExplorer />);
    await waitFor(() => expect(screen.getByTestId("target-task-select")).not.toBeDisabled());

    fireEvent.change(screen.getByTestId("target-task-select"), {
      target: { value: mockTaskList[1].task_id },
    });
    fireEvent.click(screen.getByTestId("recommend-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("recommendations-list")).toBeInTheDocument();
    });

    const explainBtns = screen.getAllByTestId("explain-btn");
    fireEvent.click(explainBtns[0]);

    await waitFor(() => {
      expect(screen.getByTestId("explanation-panel")).toBeInTheDocument();
    });
  });

  it("navigates to OrcaLab experiments with task_id prefill when Apply Transfer is clicked", async () => {
    render(<TransferExplorer />);
    await waitFor(() => expect(screen.getByTestId("target-task-select")).not.toBeDisabled());

    fireEvent.change(screen.getByTestId("target-task-select"), {
      target: { value: mockTaskList[1].task_id },
    });
    fireEvent.click(screen.getByTestId("recommend-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("recommendations-list")).toBeInTheDocument();
    });

    const applyBtns = screen.getAllByTestId("apply-btn");
    fireEvent.click(applyBtns[0]);

    expect(mockNavigate).toHaveBeenCalledWith(
      expect.stringContaining(mockTaskList[1].task_id),
    );
  });

  it("disables recommend button when no target task is selected", () => {
    render(<TransferExplorer />);
    expect(screen.getByTestId("recommend-btn")).toBeDisabled();
  });
});
