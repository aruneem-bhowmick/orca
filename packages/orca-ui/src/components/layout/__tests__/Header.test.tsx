import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { Header } from "@/components/layout/Header";
import { useAuthStore } from "@/store/auth";
import { useThemeStore } from "@/store/theme";
import { mockUser } from "@/test/mocks/handlers";

vi.mock("@/api/auth", () => ({
  getMe: vi.fn().mockRejectedValue(new Error("No session")),
}));

describe("Header", () => {
  beforeEach(() => {
    useAuthStore.getState().setAuth(mockUser, "test-token");
    // Reset to light mode before each test.
    useThemeStore.getState().setMode("light");
    document.documentElement.classList.remove("dark");
  });

  it("renders the header with breadcrumbs", () => {
    render(<Header />);
    expect(screen.getByTestId("header")).toBeInTheDocument();
    expect(screen.getByTestId("breadcrumbs")).toBeInTheDocument();
  });

  it("renders the search input", () => {
    render(<Header />);
    expect(screen.getByTestId("search-input")).toBeInTheDocument();
  });

  it("renders the notifications button with badge", () => {
    render(<Header />);
    expect(screen.getByTestId("notifications-button")).toBeInTheDocument();
    expect(screen.getByTestId("notification-badge")).toBeInTheDocument();
  });

  it("toggles dark mode via the Zustand theme store when button is clicked", () => {
    render(<Header />);

    const toggle = screen.getByTestId("dark-mode-toggle");
    expect(document.documentElement.classList.contains("dark")).toBe(false);

    fireEvent.click(toggle);
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(useThemeStore.getState().mode).toBe("dark");

    fireEvent.click(toggle);
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    expect(useThemeStore.getState().mode).toBe("light");
  });

  it("renders user email when authenticated", () => {
    render(<Header />);
    expect(screen.getByText("test@example.com")).toBeInTheDocument();
  });

  it("renders a hamburger button when onMenuToggle is provided", () => {
    const onMenuToggle = vi.fn();
    render(<Header onMenuToggle={onMenuToggle} />);
    const hamburger = screen.getByTestId("hamburger-button");
    expect(hamburger).toBeInTheDocument();
    fireEvent.click(hamburger);
    expect(onMenuToggle).toHaveBeenCalledOnce();
  });

  it("does not render a hamburger button when onMenuToggle is omitted", () => {
    render(<Header />);
    expect(screen.queryByTestId("hamburger-button")).not.toBeInTheDocument();
  });
});
