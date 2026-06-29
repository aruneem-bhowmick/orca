import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, fireEvent, within } from "@testing-library/react";
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

  it("renders top-level navigation links", () => {
    render(<Sidebar />);
    expect(screen.getByText("Dashboard")).toBeInTheDocument();
    expect(screen.getByText("History")).toBeInTheDocument();
    expect(screen.getByText("Bookmarks")).toBeInTheDocument();
  });

  it("renders navigation groups for services", () => {
    render(<Sidebar />);
    expect(screen.getByText("OrcaMind")).toBeInTheDocument();
    expect(screen.getByText("OrcaLab")).toBeInTheDocument();
    expect(screen.getByText("OrcaNet")).toBeInTheDocument();
  });

  it("renders user info in the user menu trigger", () => {
    render(<Sidebar />);
    expect(screen.getByText("testuser")).toBeInTheDocument();
    expect(screen.getByText("test@example.com")).toBeInTheDocument();
  });

  it("shows user dropdown with Profile and Sign out when user menu is clicked", () => {
    render(<Sidebar />);

    const trigger = screen.getByTestId("user-menu-trigger");
    fireEvent.click(trigger);

    expect(screen.getByTestId("user-dropdown")).toBeInTheDocument();
    expect(screen.getByText("Profile")).toBeInTheDocument();
    expect(screen.getByText("Sign out")).toBeInTheDocument();
  });

  it("collapses when toggle is clicked", () => {
    render(<Sidebar />);

    expect(screen.getByText("Dashboard")).toBeInTheDocument();

    const toggle = screen.getByTestId("sidebar-toggle");
    fireEvent.click(toggle);

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

  it("expands a nav group to show sub-items", () => {
    render(<Sidebar />);

    const orcaLabGroup = screen.getByTestId("nav-group-orcalab");
    fireEvent.click(orcaLabGroup);

    const children = screen.getByTestId("nav-children-orcalab");
    expect(children).toBeInTheDocument();
    expect(screen.getByText("Experiments")).toBeInTheDocument();
    expect(screen.getByText("Sweeps")).toBeInTheDocument();
  });

  it("does not render mobile drawer when mobileOpen is false", () => {
    render(<Sidebar mobileOpen={false} />);
    expect(screen.queryByTestId("sidebar-mobile")).not.toBeInTheDocument();
    expect(screen.queryByTestId("sidebar-backdrop")).not.toBeInTheDocument();
  });

  it("renders mobile drawer and backdrop when mobileOpen is true", () => {
    render(<Sidebar mobileOpen={true} onMobileClose={vi.fn()} />);
    expect(screen.getByTestId("sidebar-mobile")).toBeInTheDocument();
    expect(screen.getByTestId("sidebar-backdrop")).toBeInTheDocument();
  });

  it("calls onMobileClose when the mobile close button is clicked", () => {
    const onMobileClose = vi.fn();
    render(<Sidebar mobileOpen={true} onMobileClose={onMobileClose} />);
    fireEvent.click(screen.getByTestId("sidebar-mobile-close"));
    expect(onMobileClose).toHaveBeenCalledOnce();
  });

  it("calls onMobileClose when the backdrop is clicked", () => {
    const onMobileClose = vi.fn();
    render(<Sidebar mobileOpen={true} onMobileClose={onMobileClose} />);
    fireEvent.click(screen.getByTestId("sidebar-backdrop"));
    expect(onMobileClose).toHaveBeenCalledOnce();
  });

  it("calls onMobileClose when a top-level link inside sidebar-mobile is clicked", () => {
    const onMobileClose = vi.fn();
    render(<Sidebar mobileOpen={true} onMobileClose={onMobileClose} />);

    const mobileDrawer = screen.getByTestId("sidebar-mobile");
    const dashboardLink = within(mobileDrawer).getByText("Dashboard");
    fireEvent.click(dashboardLink);

    expect(onMobileClose).toHaveBeenCalledOnce();
  });

  it("calls onMobileClose when a grouped-child link under sidebar-mobile is clicked", () => {
    const onMobileClose = vi.fn();
    render(<Sidebar mobileOpen={true} onMobileClose={onMobileClose} />);

    const mobileDrawer = screen.getByTestId("sidebar-mobile");

    // First expand the group
    const orcaLabGroup = within(mobileDrawer).getByTestId("nav-group-orcalab");
    fireEvent.click(orcaLabGroup);

    // Then click the child link
    const experimentsLink = within(mobileDrawer).getByText("Experiments");
    fireEvent.click(experimentsLink);

    expect(onMobileClose).toHaveBeenCalledOnce();
  });
});
