import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { MainLayout } from "@/components/layout/MainLayout";
import { useAuthStore } from "@/store/auth";
import { mockUser } from "@/test/mocks/handlers";

vi.mock("@/api/auth", () => ({
  getMe: vi.fn().mockRejectedValue(new Error("No session")),
  logout: vi.fn().mockResolvedValue(undefined),
}));

describe("MainLayout", () => {
  beforeEach(() => {
    useAuthStore.getState().setAuth(mockUser, "test-token");
  });

  it("renders the sidebar", () => {
    render(<MainLayout />);
    expect(screen.getByTestId("sidebar")).toBeInTheDocument();
  });

  it("renders the header", () => {
    render(<MainLayout />);
    expect(screen.getByTestId("header")).toBeInTheDocument();
  });

  it("renders the main content area", () => {
    render(<MainLayout />);
    const main = document.querySelector("main");
    expect(main).toBeInTheDocument();
    expect(main).toHaveClass("flex-1", "overflow-auto", "p-6");
  });

  it("uses a full-height flex layout", () => {
    render(<MainLayout />);
    const container = screen.getByTestId("main-layout");
    expect(container).toHaveClass("flex", "h-screen", "overflow-hidden");
  });
});
