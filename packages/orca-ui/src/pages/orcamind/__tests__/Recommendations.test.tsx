import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@/test/test-utils";
import {
  RecommendationCards,
  Recommendations,
} from "@/pages/orcamind/Recommendations";
import apiClient from "@/api/client";
import { mockRecommendations } from "@/test/mocks/handlers";

vi.mock("@/api/client", () => ({
  default: {
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

/** Render the standalone Recommendations page at a given URL. */
function renderPage(url: string) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return import("@testing-library/react").then(({ render: rtlRender }) =>
    rtlRender(
      <QueryClientProvider client={qc}>
        <MemoryRouter initialEntries={[url]}>
          <Recommendations />
        </MemoryRouter>
      </QueryClientProvider>,
    ),
  );
}

describe("RecommendationCards", () => {
  beforeEach(() => vi.clearAllMocks());

  it("renders a card for each recommendation", () => {
    render(<RecommendationCards recommendations={mockRecommendations} />);
    expect(
      screen.getByTestId(`rec-card-${mockRecommendations[0].model_id}`),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId(`rec-card-${mockRecommendations[1].model_id}`),
    ).toBeInTheDocument();
  });

  it("displays model name, architecture, accuracy, and confidence", () => {
    render(<RecommendationCards recommendations={mockRecommendations} />);
    expect(screen.getByText("ResNet-50")).toBeInTheDocument();
    expect(screen.getByText("ResNet")).toBeInTheDocument();
    expect(screen.getByText("92.3%")).toBeInTheDocument();
    expect(screen.getByText("87.0%")).toBeInTheDocument();
  });

  it("assigns rank badges in order (#1, #2, ...)", () => {
    render(<RecommendationCards recommendations={mockRecommendations} />);
    expect(screen.getByText("#1")).toBeInTheDocument();
    expect(screen.getByText("#2")).toBeInTheDocument();
  });

  it("renders a Start Experiment button for each card", () => {
    render(<RecommendationCards recommendations={mockRecommendations} />);
    expect(
      screen.getByTestId(`start-experiment-${mockRecommendations[0].model_id}`),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId(`start-experiment-${mockRecommendations[1].model_id}`),
    ).toBeInTheDocument();
  });

  it("calls onStartExperiment with the recommendation when Start Experiment is clicked", () => {
    const handler = vi.fn();
    render(
      <RecommendationCards
        recommendations={mockRecommendations}
        onStartExperiment={handler}
      />,
    );
    fireEvent.click(
      screen.getByTestId(`start-experiment-${mockRecommendations[0].model_id}`),
    );
    expect(handler).toHaveBeenCalledWith(mockRecommendations[0]);
  });

  it("navigates to OrcaLab experiments when no onStartExperiment is provided", () => {
    render(<RecommendationCards recommendations={mockRecommendations} />);
    fireEvent.click(
      screen.getByTestId(`start-experiment-${mockRecommendations[0].model_id}`),
    );
    expect(mockNavigate).toHaveBeenCalledWith(
      expect.stringContaining("/dashboard/orcalab/experiments"),
    );
  });

  it("shows the no-recommendations message for an empty list", () => {
    render(<RecommendationCards recommendations={[]} />);
    expect(screen.getByTestId("no-recommendations")).toBeInTheDocument();
  });
});

describe("Recommendations page", () => {
  beforeEach(() => vi.clearAllMocks());

  it("renders the page heading", async () => {
    await renderPage("/dashboard/orcamind/recommendations");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "Recommendations",
    );
  });

  it("shows a prompt when no task_id query param is present", async () => {
    await renderPage("/dashboard/orcamind/recommendations");
    expect(screen.getByTestId("no-task-selected")).toBeInTheDocument();
  });

  it("fetches recommendations when task_id is in the query string", async () => {
    vi.mocked(apiClient.post).mockResolvedValue({ data: mockRecommendations });
    await renderPage("/dashboard/orcamind/recommendations?task_id=task-abc");
    expect(apiClient.post).toHaveBeenCalledWith(
      "/orcamind/recommend",
      expect.objectContaining({ task_id: "task-abc" }),
    );
    await waitFor(() => {
      expect(screen.getByTestId("recommendation-cards")).toBeInTheDocument();
    });
  });

  it("shows a loading state while the request is in flight", async () => {
    vi.mocked(apiClient.post).mockReturnValue(new Promise(() => {}));
    await renderPage("/dashboard/orcamind/recommendations?task_id=task-abc");
    expect(screen.getByTestId("rec-loading")).toBeInTheDocument();
  });

  it("shows an error state when the request fails", async () => {
    vi.mocked(apiClient.post).mockRejectedValue(new Error("server error"));
    await renderPage("/dashboard/orcamind/recommendations?task_id=task-abc");
    await waitFor(() => {
      expect(screen.getByTestId("rec-error")).toBeInTheDocument();
    });
  });
});
