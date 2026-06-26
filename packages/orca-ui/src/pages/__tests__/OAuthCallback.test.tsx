/**
 * Tests for the OAuthCallback page component.
 *
 * Covers: successful OAuth code exchange, error parameter display,
 * missing provider handling, and API failure handling.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { OAuthCallback } from "@/pages/OAuthCallback";
import * as authApi from "@/api/auth";
import { useAuthStore } from "@/store/auth";
import { mockUser, mockTokenResponse } from "@/test/mocks/handlers";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

vi.mock("@/api/auth");

/**
 * Render OAuthCallback with custom URL search params via MemoryRouter.
 *
 * @param searchParams - Query string to append to the /oauth/callback route.
 */
function renderWithParams(searchParams: string) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[`/oauth/callback${searchParams}`]}>
        <Routes>
          <Route path="/oauth/callback" element={<OAuthCallback />} />
          <Route path="/dashboard" element={<div>Dashboard</div>} />
          <Route path="/login" element={<div>Login Page</div>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
    { wrapper: undefined as unknown as undefined },
  );
}

describe("OAuthCallback", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useAuthStore.getState().clearAuth();
  });

  it("exchanges code and redirects to dashboard on success", async () => {
    vi.mocked(authApi.exchangeOAuthCode).mockResolvedValue(mockTokenResponse);
    vi.mocked(authApi.getMe).mockResolvedValue(mockUser);

    renderWithParams("?provider=google&code=auth-code-123");

    await waitFor(() => {
      expect(screen.getByText("Dashboard")).toBeInTheDocument();
    });

    expect(authApi.exchangeOAuthCode).toHaveBeenCalledWith(
      "google",
      expect.any(URLSearchParams),
    );
    expect(authApi.getMe).toHaveBeenCalled();
    expect(useAuthStore.getState().isAuthenticated).toBe(true);
    expect(useAuthStore.getState().user).toEqual(mockUser);
  });

  it("displays error when error parameter is present in URL", async () => {
    renderWithParams("?error=access_denied&provider=github");

    await waitFor(() => {
      expect(screen.getByTestId("oauth-error")).toHaveTextContent("access_denied");
    });

    expect(screen.getByText("Back to login")).toBeInTheDocument();
    expect(authApi.exchangeOAuthCode).not.toHaveBeenCalled();
  });

  it("displays error when provider parameter is missing", async () => {
    renderWithParams("?code=auth-code-123");

    await waitFor(() => {
      expect(screen.getByTestId("oauth-error")).toHaveTextContent("Missing OAuth provider.");
    });

    expect(authApi.exchangeOAuthCode).not.toHaveBeenCalled();
  });

  it("displays error when code parameter is missing", async () => {
    renderWithParams("?provider=google");

    await waitFor(() => {
      expect(screen.getByTestId("oauth-error")).toHaveTextContent("Missing authorization code.");
    });

    expect(authApi.exchangeOAuthCode).not.toHaveBeenCalled();
  });

  it("displays error and clears auth on API failure", async () => {
    vi.mocked(authApi.exchangeOAuthCode).mockRejectedValue(
      new Error("Network error"),
    );

    renderWithParams("?provider=github&code=bad-code");

    await waitFor(() => {
      expect(screen.getByTestId("oauth-error")).toHaveTextContent(
        "Failed to complete authentication.",
      );
    });

    expect(useAuthStore.getState().isAuthenticated).toBe(false);
  });

  it("shows a loading spinner while processing", () => {
    vi.mocked(authApi.exchangeOAuthCode).mockReturnValue(
      new Promise(() => {}),
    );

    renderWithParams("?provider=google&code=auth-code-123");

    const spinner = document.querySelector(".animate-spin");
    expect(spinner).toBeInTheDocument();
  });
});
