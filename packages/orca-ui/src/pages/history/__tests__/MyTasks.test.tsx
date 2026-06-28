import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { MyTasks } from "@/pages/history/MyTasks";
import apiClient from "@/api/client";
import { mockTaskActivityPage, mockActivityEntry } from "@/test/mocks/handlers";

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
    if (url === "/history/tasks") return Promise.resolve({ data: mockTaskActivityPage });
    return Promise.reject(new Error(`Unexpected GET: ${url}`));
  });
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("MyTasks", () => {
  it("renders the page heading", () => {
    render(<MyTasks />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("My Tasks");
  });

  it("shows a loading state before data arrives", () => {
    vi.mocked(apiClient.get).mockReturnValue(new Promise(() => {}));
    render(<MyTasks />);
    expect(screen.getByTestId("my-tasks-loading")).toBeInTheDocument();
  });

  it("renders the task entry list after data loads", async () => {
    render(<MyTasks />);
    await waitFor(() => {
      expect(screen.getByTestId("my-tasks-list")).toBeInTheDocument();
    });
  });

  it("renders an entry row for each task activity entry", async () => {
    render(<MyTasks />);
    await waitFor(() => {
      expect(
        screen.getByTestId(`task-entry-${mockActivityEntry.id}`),
      ).toBeInTheDocument();
    });
  });

  it("displays the action text for each entry", async () => {
    render(<MyTasks />);
    await waitFor(() => {
      expect(screen.getByText(mockActivityEntry.action)).toBeInTheDocument();
    });
  });

  it("shows an error state when the API fails", async () => {
    vi.mocked(apiClient.get).mockRejectedValue(new Error("network"));
    render(<MyTasks />);
    await waitFor(() => {
      expect(screen.getByTestId("my-tasks-error")).toBeInTheDocument();
    });
  });

  it("shows an empty state when no task activity is returned", async () => {
    vi.mocked(apiClient.get).mockResolvedValue({
      data: { items: [], total: 0, page: 1, per_page: 20, pages: 0 },
    });
    render(<MyTasks />);
    await waitFor(() => {
      expect(screen.getByTestId("my-tasks-empty")).toBeInTheDocument();
    });
  });

  it("navigates to OrcaMind task detail when an entry with a resource_id is clicked", async () => {
    render(<MyTasks />);
    await waitFor(() => {
      expect(screen.getByTestId(`task-entry-${mockActivityEntry.id}`)).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId(`task-entry-${mockActivityEntry.id}`));
    expect(mockNavigate).toHaveBeenCalledWith(
      expect.stringContaining(mockActivityEntry.resource_id!),
    );
  });

  it("navigates on Enter key press on an entry", async () => {
    render(<MyTasks />);
    await waitFor(() => {
      expect(screen.getByTestId(`task-entry-${mockActivityEntry.id}`)).toBeInTheDocument();
    });
    fireEvent.keyDown(screen.getByTestId(`task-entry-${mockActivityEntry.id}`), {
      key: "Enter",
    });
    expect(mockNavigate).toHaveBeenCalledWith(
      expect.stringContaining(mockActivityEntry.resource_id!),
    );
  });
});
