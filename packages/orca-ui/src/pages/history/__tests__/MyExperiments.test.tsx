import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { MyExperiments } from "@/pages/history/MyExperiments";
import apiClient from "@/api/client";
import { mockExpActivityPage, mockExpActivityEntry } from "@/test/mocks/handlers";

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
  vi.mocked(apiClient.get).mockImplementation((url: string) => {
    if (url === "/history/experiments")
      return Promise.resolve({ data: mockExpActivityPage });
    return Promise.reject(new Error(`Unexpected GET: ${url}`));
  });
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("MyExperiments", () => {
  it("renders the page heading", () => {
    render(<MyExperiments />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("My Experiments");
  });

  it("shows a loading state before data arrives", () => {
    vi.mocked(apiClient.get).mockReturnValue(new Promise(() => {}));
    render(<MyExperiments />);
    expect(screen.getByTestId("my-exp-loading")).toBeInTheDocument();
  });

  it("renders the experiment entry list after data loads", async () => {
    render(<MyExperiments />);
    await waitFor(() => {
      expect(screen.getByTestId("my-exp-list")).toBeInTheDocument();
    });
  });

  it("renders an entry row for each experiment activity entry", async () => {
    render(<MyExperiments />);
    await waitFor(() => {
      expect(
        screen.getByTestId(`exp-entry-${mockExpActivityEntry.id}`),
      ).toBeInTheDocument();
    });
  });

  it("shows the action text for each entry", async () => {
    render(<MyExperiments />);
    await waitFor(() => {
      expect(screen.getByText(mockExpActivityEntry.action)).toBeInTheDocument();
    });
  });

  it("renders a status badge derived from the action string", async () => {
    render(<MyExperiments />);
    await waitFor(() => {
      // "experiment_completed" should produce a 'completed' badge
      expect(screen.getByTestId("exp-status-badge-completed")).toBeInTheDocument();
    });
  });

  it("shows an error state when the API fails", async () => {
    vi.mocked(apiClient.get).mockRejectedValue(new Error("network"));
    render(<MyExperiments />);
    await waitFor(() => {
      expect(screen.getByTestId("my-exp-error")).toBeInTheDocument();
    });
  });

  it("shows an empty state when no experiment activity is returned", async () => {
    vi.mocked(apiClient.get).mockResolvedValue({
      data: { items: [], total: 0, page: 1, per_page: 20, pages: 0 },
    });
    render(<MyExperiments />);
    await waitFor(() => {
      expect(screen.getByTestId("my-exp-empty")).toBeInTheDocument();
    });
  });

  it("navigates to OrcaLab experiment detail when an entry with a resource_id is clicked", async () => {
    render(<MyExperiments />);
    await waitFor(() => {
      expect(
        screen.getByTestId(`exp-entry-${mockExpActivityEntry.id}`),
      ).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId(`exp-entry-${mockExpActivityEntry.id}`));
    expect(mockNavigate).toHaveBeenCalledWith(
      expect.stringContaining(mockExpActivityEntry.resource_id!),
    );
  });

  it("navigates on Enter key press on an entry", async () => {
    render(<MyExperiments />);
    await waitFor(() => {
      expect(
        screen.getByTestId(`exp-entry-${mockExpActivityEntry.id}`),
      ).toBeInTheDocument();
    });
    fireEvent.keyDown(screen.getByTestId(`exp-entry-${mockExpActivityEntry.id}`), {
      key: "Enter",
    });
    expect(mockNavigate).toHaveBeenCalledWith(
      expect.stringContaining(mockExpActivityEntry.resource_id!),
    );
  });
});
