import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { RetrievalView } from "@/pages/orcanet/RetrievalView";
import apiClient from "@/api/client";
import { mockRetrieveResults } from "@/test/mocks/handlers";

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
// Setup
// ---------------------------------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(apiClient.post).mockImplementation((url: string) => {
    if (url === "/orcanet/retrieve") return Promise.resolve({ data: mockRetrieveResults });
    return Promise.reject(new Error(`Unexpected POST: ${url}`));
  });
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("RetrievalView", () => {
  it("renders the page heading", () => {
    render(<RetrievalView />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Task Retrieval");
  });

  it("renders the query input and search button", () => {
    render(<RetrievalView />);
    expect(screen.getByTestId("retrieval-query-input")).toBeInTheDocument();
    expect(screen.getByTestId("retrieval-search-btn")).toBeInTheDocument();
  });

  it("disables search button when query is empty", () => {
    render(<RetrievalView />);
    expect(screen.getByTestId("retrieval-search-btn")).toBeDisabled();
  });

  it("enables search button when query has text", () => {
    render(<RetrievalView />);
    fireEvent.change(screen.getByTestId("retrieval-query-input"), {
      target: { value: "image classification" },
    });
    expect(screen.getByTestId("retrieval-search-btn")).not.toBeDisabled();
  });

  it("calls POST /orcanet/retrieve with the query on submit", async () => {
    render(<RetrievalView />);
    fireEvent.change(screen.getByTestId("retrieval-query-input"), {
      target: { value: "medical X-ray classification" },
    });
    fireEvent.click(screen.getByTestId("retrieval-search-btn"));

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        "/orcanet/retrieve",
        expect.objectContaining({ query: "medical X-ray classification" }),
      );
    });
  });

  it("renders result cards after a successful search", async () => {
    render(<RetrievalView />);
    fireEvent.change(screen.getByTestId("retrieval-query-input"), {
      target: { value: "vision" },
    });
    fireEvent.click(screen.getByTestId("retrieval-search-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("retrieval-results")).toBeInTheDocument();
      expect(
        screen.getByTestId(`result-card-${mockRetrieveResults[0].task_id}`),
      ).toBeInTheDocument();
      expect(
        screen.getByTestId(`result-card-${mockRetrieveResults[1].task_id}`),
      ).toBeInTheDocument();
    });
  });

  it("displays similarity score percentages in result cards", async () => {
    render(<RetrievalView />);
    fireEvent.change(screen.getByTestId("retrieval-query-input"), {
      target: { value: "vision" },
    });
    fireEvent.click(screen.getByTestId("retrieval-search-btn"));

    await waitFor(() => {
      const scores = screen.getAllByTestId("result-score");
      expect(scores[0]).toHaveTextContent("95%");
      expect(scores[1]).toHaveTextContent("78%");
    });
  });

  it("shows the domain for each result card", async () => {
    render(<RetrievalView />);
    fireEvent.change(screen.getByTestId("retrieval-query-input"), {
      target: { value: "vision" },
    });
    fireEvent.click(screen.getByTestId("retrieval-search-btn"));

    await waitFor(() => {
      const domains = screen.getAllByTestId("result-domain");
      domains.forEach((d) => expect(d).toHaveTextContent("vision"));
    });
  });

  it("shows empty state when search returns no results", async () => {
    vi.mocked(apiClient.post).mockResolvedValue({ data: [] });
    render(<RetrievalView />);
    fireEvent.change(screen.getByTestId("retrieval-query-input"), {
      target: { value: "nothing" },
    });
    fireEvent.click(screen.getByTestId("retrieval-search-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("retrieval-empty")).toBeInTheDocument();
    });
  });

  it("shows an error when POST /orcanet/retrieve fails", async () => {
    vi.mocked(apiClient.post).mockRejectedValue(new Error("network"));
    render(<RetrievalView />);
    fireEvent.change(screen.getByTestId("retrieval-query-input"), {
      target: { value: "test" },
    });
    fireEvent.click(screen.getByTestId("retrieval-search-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("retrieval-error")).toBeInTheDocument();
    });
  });

  it("navigates to OrcaMind task detail when View Details is clicked", async () => {
    render(<RetrievalView />);
    fireEvent.change(screen.getByTestId("retrieval-query-input"), {
      target: { value: "vision" },
    });
    fireEvent.click(screen.getByTestId("retrieval-search-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("retrieval-results")).toBeInTheDocument();
    });

    const viewBtns = screen.getAllByTestId("view-task-btn");
    fireEvent.click(viewBtns[0]);

    expect(mockNavigate).toHaveBeenCalledWith(
      expect.stringContaining(mockRetrieveResults[0].task_id),
    );
  });

  it("submits search on form Enter key", async () => {
    render(<RetrievalView />);
    const input = screen.getByTestId("retrieval-query-input");
    fireEvent.change(input, { target: { value: "nlp task" } });
    fireEvent.keyDown(input, { key: "Enter", code: "Enter" });

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        "/orcanet/retrieve",
        expect.objectContaining({ query: "nlp task" }),
      );
    });
  });
});
