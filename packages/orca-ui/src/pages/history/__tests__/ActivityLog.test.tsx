import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { ActivityLog } from "@/pages/history/ActivityLog";
import apiClient from "@/api/client";
import { mockActivityPage, mockActivityEntry } from "@/test/mocks/handlers";
import type { PaginatedResponse, ActivityLogEntry } from "@/api/types";

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
// Multi-page mock for infinite scroll tests
// ---------------------------------------------------------------------------

/** Two activity entries for pagination tests. */
const entry2: ActivityLogEntry = {
  id: "log-page2-001",
  user_id: "user-001",
  action: "experiment_started",
  resource_type: "experiment",
  resource_id: "exp-001",
  service: "orcalab",
  details: null,
  created_at: "2024-03-16T10:00:00Z",
};

const page2: PaginatedResponse<ActivityLogEntry> = {
  items: [entry2],
  total: 2,
  page: 2,
  per_page: 20,
  pages: 2,
};

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(apiClient.get).mockImplementation((url: string, config?: { params?: Record<string, unknown> }) => {
    if (url === "/history") {
      const page = config?.params?.page ?? 1;
      return Promise.resolve({ data: page === 1 ? mockActivityPage : page2 });
    }
    return Promise.reject(new Error(`Unexpected GET: ${url}`));
  });
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ActivityLog", () => {
  it("renders the page heading", () => {
    render(<ActivityLog />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Activity Log");
  });

  it("renders the service filter dropdown", () => {
    render(<ActivityLog />);
    expect(screen.getByTestId("service-filter")).toBeInTheDocument();
  });

  it("renders the date range inputs", () => {
    render(<ActivityLog />);
    expect(screen.getByTestId("date-from")).toBeInTheDocument();
    expect(screen.getByTestId("date-to")).toBeInTheDocument();
  });

  it("shows a loading state before data arrives", () => {
    vi.mocked(apiClient.get).mockReturnValue(new Promise(() => {}));
    render(<ActivityLog />);
    expect(screen.getByTestId("activity-loading")).toBeInTheDocument();
  });

  it("renders the timeline after data loads", async () => {
    render(<ActivityLog />);
    await waitFor(() => {
      expect(screen.getByTestId("activity-timeline")).toBeInTheDocument();
    });
  });

  it("renders a timeline entry for each loaded item", async () => {
    render(<ActivityLog />);
    await waitFor(() => {
      expect(
        screen.getByTestId(`activity-entry-${mockActivityEntry.id}`),
      ).toBeInTheDocument();
    });
  });

  it("shows the action text for each entry", async () => {
    render(<ActivityLog />);
    await waitFor(() => {
      expect(screen.getByText(mockActivityEntry.action)).toBeInTheDocument();
    });
  });

  it("shows a service badge for entries with a service", async () => {
    render(<ActivityLog />);
    await waitFor(() => {
      expect(
        screen.getByTestId(`service-badge-${mockActivityEntry.service}`),
      ).toBeInTheDocument();
    });
  });

  it("shows an error state when the API fails", async () => {
    vi.mocked(apiClient.get).mockRejectedValue(new Error("network"));
    render(<ActivityLog />);
    await waitFor(() => {
      expect(screen.getByTestId("activity-error")).toBeInTheDocument();
    });
  });

  it("shows an empty state when no entries are returned", async () => {
    vi.mocked(apiClient.get).mockResolvedValue({
      data: { items: [], total: 0, page: 1, per_page: 20, pages: 0 },
    });
    render(<ActivityLog />);
    await waitFor(() => {
      expect(screen.getByTestId("activity-empty")).toBeInTheDocument();
    });
  });

  it("refetches with service filter when dropdown changes", async () => {
    render(<ActivityLog />);
    await waitFor(() => {
      expect(screen.getByTestId("activity-timeline")).toBeInTheDocument();
    });
    fireEvent.change(screen.getByTestId("service-filter"), {
      target: { value: "orcamind" },
    });
    await waitFor(() => {
      expect(apiClient.get).toHaveBeenCalledWith(
        "/history",
        expect.objectContaining({ params: expect.objectContaining({ service: "orcamind" }) }),
      );
    });
  });

  it("renders a scroll sentinel element", async () => {
    render(<ActivityLog />);
    expect(screen.getByTestId("scroll-sentinel")).toBeInTheDocument();
  });

  it("shows end-of-log message on a single-page response", async () => {
    render(<ActivityLog />);
    await waitFor(() => {
      expect(screen.getByTestId("end-of-log")).toBeInTheDocument();
    });
  });

  it("hides entries before the date-from filter", async () => {
    render(<ActivityLog />);
    await waitFor(() => {
      expect(screen.getByTestId("activity-timeline")).toBeInTheDocument();
    });
    // Entry created_at is 2024-03-15; a from-date of 2024-03-17 should exclude it.
    fireEvent.change(screen.getByTestId("date-from"), {
      target: { value: "2024-03-17" },
    });
    await waitFor(() => {
      expect(
        screen.queryByTestId(`activity-entry-${mockActivityEntry.id}`),
      ).not.toBeInTheDocument();
      expect(screen.getByTestId("activity-empty")).toBeInTheDocument();
    });
  });

  it("hides entries after the date-to filter", async () => {
    render(<ActivityLog />);
    await waitFor(() => {
      expect(screen.getByTestId("activity-timeline")).toBeInTheDocument();
    });
    // Entry created_at is 2024-03-15; a to-date of 2024-03-13 should exclude it.
    fireEvent.change(screen.getByTestId("date-to"), {
      target: { value: "2024-03-13" },
    });
    await waitFor(() => {
      expect(
        screen.queryByTestId(`activity-entry-${mockActivityEntry.id}`),
      ).not.toBeInTheDocument();
      expect(screen.getByTestId("activity-empty")).toBeInTheDocument();
    });
  });

  it("fetches the next page when the scroll sentinel becomes visible", async () => {
    vi.mocked(apiClient.get).mockResolvedValueOnce({
      data: {
        items: [mockActivityEntry],
        total: 2,
        page: 1,
        per_page: 1,
        pages: 2,
      },
    });

    render(<ActivityLog />);
    
    await waitFor(() => {
      expect(screen.getByTestId(`activity-entry-${mockActivityEntry.id}`)).toBeInTheDocument();
    });

    const nextPageEntry = { ...mockActivityEntry, id: 999, action: "next_page_action" };
    vi.mocked(apiClient.get).mockResolvedValueOnce({
      data: {
        items: [nextPageEntry],
        total: 2,
        page: 2,
        per_page: 1,
        pages: 2,
      },
    });

    const observers = (global.IntersectionObserver as any).observers;
    expect(observers.length).toBeGreaterThan(0);
    const lastObserver = observers[observers.length - 1];

    const sentinel = screen.getByTestId("scroll-sentinel");
    lastObserver.trigger(true, sentinel);

    await waitFor(() => {
      expect(screen.getByTestId("activity-entry-999")).toBeInTheDocument();
    });
  });
});
