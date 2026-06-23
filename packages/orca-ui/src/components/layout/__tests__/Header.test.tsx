import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, fireEvent } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { Header } from "@/components/layout/Header";
import { useAuthStore } from "@/store/auth";
import { mockUser } from "@/test/mocks/handlers";

vi.mock("@/api/auth", () => ({
  getMe: vi.fn().mockRejectedValue(new Error("No session")),
}));

describe("Header", () => {
  beforeEach(() => {
    useAuthStore.getState().setAuth(mockUser, "test-token");
    document.documentElement.classList.remove("dark");
  });

  it("renders the header", () => {
    render(<Header />);
    expect(screen.getByTestId("header")).toBeInTheDocument();
    expect(screen.getByText("Orca Platform")).toBeInTheDocument();
  });

  it("toggles dark mode when button is clicked", () => {
    render(<Header />);

    const toggle = screen.getByTestId("dark-mode-toggle");
    expect(document.documentElement.classList.contains("dark")).toBe(false);

    fireEvent.click(toggle);
    expect(document.documentElement.classList.contains("dark")).toBe(true);

    fireEvent.click(toggle);
    expect(document.documentElement.classList.contains("dark")).toBe(false);
  });

  it("renders user email when authenticated", () => {
    render(<Header />);
    expect(screen.getByText("test@example.com")).toBeInTheDocument();
  });
});
