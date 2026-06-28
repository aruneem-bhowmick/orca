import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { TaskList } from "@/pages/orcamind/TaskList";
import apiClient from "@/api/client";
import { mockTaskList, mockTask } from "@/test/mocks/handlers";

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

describe("TaskList", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(apiClient.get).mockImplementation((url: string) => {
      if (url === "/orcamind/tasks") {
        return Promise.resolve({ data: mockTaskList });
      }
      return Promise.reject(new Error(`Unexpected: ${url}`));
    });
  });

  it("renders the page heading", () => {
    render(<TaskList />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "Tasks",
    );
  });

  it("shows a skeleton loading state before tasks arrive", () => {
    vi.mocked(apiClient.get).mockReturnValue(new Promise(() => {}));
    render(<TaskList />);
    expect(screen.getByTestId("table-skeleton")).toBeInTheDocument();
  });

  it("renders the task table when data loads", async () => {
    render(<TaskList />);
    await waitFor(() => {
      expect(screen.getByTestId("task-table")).toBeInTheDocument();
    });
  });

  it("displays a row for each task", async () => {
    render(<TaskList />);
    await waitFor(() => {
      expect(
        screen.getByTestId(`task-row-${mockTaskList[0].task_id}`),
      ).toBeInTheDocument();
      expect(
        screen.getByTestId(`task-row-${mockTaskList[1].task_id}`),
      ).toBeInTheDocument();
    });
  });

  it("renders task name, domain, and type in each row", async () => {
    render(<TaskList />);
    await waitFor(() => {
      expect(screen.getByText("Image Classification")).toBeInTheDocument();
    });
    expect(screen.getByText("vision")).toBeInTheDocument();
    expect(screen.getByText("Sentiment Analysis")).toBeInTheDocument();
    expect(screen.getByText("nlp")).toBeInTheDocument();
  });

  it("shows empty state when no tasks are returned", async () => {
    vi.mocked(apiClient.get).mockResolvedValue({ data: [] });
    render(<TaskList />);
    await waitFor(() => {
      expect(screen.getByTestId("task-list-empty")).toBeInTheDocument();
    });
  });

  it("shows an error state when the API call fails", async () => {
    vi.mocked(apiClient.get).mockRejectedValue(new Error("network error"));
    render(<TaskList />);
    await waitFor(() => {
      expect(screen.getByTestId("task-list-error")).toBeInTheDocument();
    });
  });

  it("filters rows by the search input", async () => {
    render(<TaskList />);
    await waitFor(() => {
      expect(screen.getByTestId("task-table")).toBeInTheDocument();
    });
    const search = screen.getByTestId("task-search");
    fireEvent.change(search, { target: { value: "nlp" } });
    await waitFor(() => {
      expect(screen.queryByText("Image Classification")).not.toBeInTheDocument();
      expect(screen.getByText("Sentiment Analysis")).toBeInTheDocument();
    });
  });

  it("shows all tasks when search is cleared", async () => {
    render(<TaskList />);
    await waitFor(() => {
      expect(screen.getByTestId("task-table")).toBeInTheDocument();
    });
    const search = screen.getByTestId("task-search");
    fireEvent.change(search, { target: { value: "nlp" } });
    fireEvent.change(search, { target: { value: "" } });
    await waitFor(() => {
      expect(screen.getByText("Image Classification")).toBeInTheDocument();
      expect(screen.getByText("Sentiment Analysis")).toBeInTheDocument();
    });
  });

  it("navigates to task detail when a row is clicked", async () => {
    render(<TaskList />);
    await waitFor(() => {
      expect(
        screen.getByTestId(`task-row-${mockTaskList[0].task_id}`),
      ).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId(`task-row-${mockTaskList[0].task_id}`));
    expect(mockNavigate).toHaveBeenCalledWith(
      expect.stringContaining(mockTaskList[0].task_id),
    );
  });

  it("renders the Embed New Task button", () => {
    render(<TaskList />);
    expect(screen.getByTestId("embed-task-btn")).toBeInTheDocument();
  });

  it("opens the embed dialog when the Embed New Task button is clicked", async () => {
    render(<TaskList />);
    fireEvent.click(screen.getByTestId("embed-task-btn"));
    expect(screen.getByTestId("embed-dialog")).toBeInTheDocument();
  });

  it("closes the embed dialog when Cancel is clicked", async () => {
    render(<TaskList />);
    fireEvent.click(screen.getByTestId("embed-task-btn"));
    expect(screen.getByTestId("embed-dialog")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(screen.queryByTestId("embed-dialog")).not.toBeInTheDocument();
  });

  it("submits the embed form with valid data and closes on success", async () => {
    vi.mocked(apiClient.post).mockResolvedValue({ data: mockTask });
    render(<TaskList />);
    fireEvent.click(screen.getByTestId("embed-task-btn"));

    fireEvent.change(screen.getByTestId("embed-name"), {
      target: { value: "New Task" },
    });
    fireEvent.change(screen.getByTestId("embed-domain"), {
      target: { value: "vision" },
    });
    fireEvent.change(screen.getByTestId("embed-task-type"), {
      target: { value: "classification" },
    });
    fireEvent.change(screen.getByTestId("embed-n-samples"), {
      target: { value: "1000" },
    });

    fireEvent.click(screen.getByRole("button", { name: /embed task/i }));

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        "/orcamind/tasks",
        expect.objectContaining({ name: "New Task", domain: "vision" }),
      );
    });
    await waitFor(() => {
      expect(screen.queryByTestId("embed-dialog")).not.toBeInTheDocument();
    });
  });

  it("shows validation errors when the form is submitted empty", async () => {
    render(<TaskList />);
    fireEvent.click(screen.getByTestId("embed-task-btn"));
    fireEvent.click(screen.getByRole("button", { name: /embed task/i }));
    expect(screen.getByText("Name is required")).toBeInTheDocument();
    expect(screen.getByText("Domain is required")).toBeInTheDocument();
  });

  it("sorts tasks by domain when the Domain column header is clicked", async () => {
    render(<TaskList />);
    await waitFor(() => {
      expect(screen.getByTestId("task-table")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByTestId("sort-domain"));
    const rows = screen.getAllByRole("row").slice(1);
    const firstDomain = rows[0].querySelectorAll("td")[1].textContent;
    expect(firstDomain).toBe("nlp");
  });
});
