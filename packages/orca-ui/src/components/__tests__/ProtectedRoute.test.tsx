import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { useAuthStore } from "@/store/auth";
import { mockUser } from "@/test/mocks/handlers";

vi.mock("@/api/auth", () => ({
  getMe: vi.fn().mockRejectedValue(new Error("No session")),
}));

describe("ProtectedRoute", () => {
  beforeEach(() => {
    useAuthStore.getState().clearAuth();
  });

  it("shows loading spinner during auth check", () => {
    render(
      <ProtectedRoute>
        <div>Protected content</div>
      </ProtectedRoute>,
    );
    expect(screen.getByTestId("auth-loading")).toBeInTheDocument();
  });

  it("renders children when authenticated after loading", async () => {
    useAuthStore.getState().setAuth(mockUser, "valid-token");

    render(
      <ProtectedRoute>
        <div data-testid="protected-content">Protected content</div>
      </ProtectedRoute>,
    );

    await waitFor(() => {
      expect(screen.getByTestId("protected-content")).toBeInTheDocument();
    });
  });

  it("redirects to login when not authenticated after loading", async () => {
    render(
      <ProtectedRoute>
        <div data-testid="protected-content">Protected content</div>
      </ProtectedRoute>,
    );

    await waitFor(() => {
      expect(screen.queryByTestId("auth-loading")).not.toBeInTheDocument();
    });

    expect(screen.queryByTestId("protected-content")).not.toBeInTheDocument();
  });
});
