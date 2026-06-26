import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { App } from "@/App";
import { useAuthStore } from "@/store/auth";

vi.mock("@/api/auth", () => ({
  getMe: vi.fn().mockRejectedValue(new Error("No session")),
}));

vi.mock("@/api/client", () => ({
  default: {
    get: vi.fn().mockImplementation((url: string) => {
      if (url === "/health") {
        return Promise.resolve({
          data: {
            status: "healthy",
            services: {
              postgres: true,
              redis: true,
              orcamind: true,
              orcalab: true,
              orcanet: true,
            },
          },
        });
      }
      if (url === "/dashboard/stats") {
        return Promise.resolve({
          data: {
            tasks_registered: 42,
            experiments_run: 128,
            transfers_scored: 56,
          },
        });
      }
      return Promise.resolve({ data: {} });
    }),
    interceptors: {
      request: { use: vi.fn(), handlers: [] },
      response: { use: vi.fn(), handlers: [] },
    },
  },
}));

describe("App", () => {
  beforeEach(() => {
    useAuthStore.getState().clearAuth();
    window.history.pushState({}, "", "/");
  });

  it("renders the landing page at /", () => {
    render(<App />);
    expect(
      screen.getByText("Orca: Meta-Learning Platform"),
    ).toBeInTheDocument();
  });

  it("renders the login page at /login", () => {
    window.history.pushState({}, "", "/login");
    render(<App />);
    expect(screen.getByText("Sign in to Orca")).toBeInTheDocument();
  });

  it("renders the register page at /register", () => {
    window.history.pushState({}, "", "/register");
    render(<App />);
    expect(screen.getByText("Create your Orca account")).toBeInTheDocument();
  });

  it("shows loading spinner for protected routes when unauthenticated", () => {
    window.history.pushState({}, "", "/dashboard");
    render(<App />);
    expect(screen.getByTestId("auth-loading")).toBeInTheDocument();
  });
});
