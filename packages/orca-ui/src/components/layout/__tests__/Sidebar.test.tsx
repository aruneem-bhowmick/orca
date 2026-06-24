import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { Sidebar } from "@/components/layout/Sidebar";
import { useAuthStore } from "@/store/auth";
import { mockUser } from "@/test/mocks/handlers";

vi.mock("@/api/auth", () => ({
  getMe: vi.fn().mockRejectedValue(new Error("No session")),
  logout: vi.fn().mockResolvedValue(undefined),
}));

describe("Sidebar", () => {
  beforeEach(() => {
    useAuthStore.getState().setAuth(mockUser, "test-token");
  });

  it("renders navigation links", () => {
    render(<Sidebar />);
    expect(screen.getByText("Dashboard")).toBeInTheDocument();
    expect(screen.getByText("Tasks")).toBeInTheDocument();
    expect(screen.getByText("Experiments")).toBeInTheDocument();
    expect(screen.getByText("Sweeps")).toBeInTheDocument();
    expect(screen.getByText("Transfers")).toBeInTheDocument();
    expect(screen.getByText("History")).toBeInTheDocument();
    expect(screen.getByText("Bookmarks")).toBeInTheDocument();
  });

  it("renders user info", () => {
    render(<Sidebar />);
    expect(screen.getByText("testuser")).toBeInTheDocument();
    expect(screen.getByText("Sign out")).toBeInTheDocument();
  });

  it("collapses when toggle is clicked", () => {
    render(<Sidebar />);

    // Before collapse, nav labels should be visible
    expect(screen.getByText("Dashboard")).toBeInTheDocument();

    const toggle = screen.getByTestId("sidebar-toggle");
    fireEvent.click(toggle);

    // After collapse, text labels should be hidden
    expect(screen.queryByText("Dashboard")).not.toBeInTheDocument();
  });

  it("expands when toggle is clicked again", () => {
    render(<Sidebar />);

    const toggle = screen.getByTestId("sidebar-toggle");
    fireEvent.click(toggle); // collapse
    fireEvent.click(toggle); // expand

    expect(screen.getByText("Dashboard")).toBeInTheDocument();
  });

  it("renders Orca brand text when expanded", () => {
    render(<Sidebar />);
    expect(screen.getByText("Orca")).toBeInTheDocument();
  });
});
