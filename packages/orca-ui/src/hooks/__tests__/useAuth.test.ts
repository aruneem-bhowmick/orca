import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { useAuth } from "@/hooks/useAuth";
import { useAuthStore } from "@/store/auth";
import * as authApi from "@/api/auth";
import { mockUser, mockTokenResponse } from "@/test/mocks/handlers";

vi.mock("@/api/auth", () => ({
  login: vi.fn(),
  register: vi.fn(),
  logout: vi.fn(),
  getMe: vi.fn(),
}));

describe("useAuth", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useAuthStore.getState().clearAuth();
  });

  it("starts in loading state and resolves after session check", async () => {
    vi.mocked(authApi.getMe).mockRejectedValueOnce(new Error("No session"));

    const { result } = renderHook(() => useAuth());
    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });
    expect(result.current.isAuthenticated).toBe(false);
  });

  it("restores session when getMe succeeds and token exists", async () => {
    useAuthStore.getState().setAuth(mockUser, "existing-token");
    vi.mocked(authApi.getMe).mockResolvedValueOnce(mockUser);

    const { result } = renderHook(() => useAuth());

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });
    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user).toEqual(mockUser);
  });

  it("login calls API and sets auth state", async () => {
    vi.mocked(authApi.getMe).mockRejectedValueOnce(new Error("No session"));
    vi.mocked(authApi.login).mockResolvedValueOnce(mockTokenResponse);
    vi.mocked(authApi.getMe).mockResolvedValueOnce(mockUser);

    const { result } = renderHook(() => useAuth());

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    await result.current.login({ email: "test@example.com", password: "pass" });

    expect(authApi.login).toHaveBeenCalledWith({
      email: "test@example.com",
      password: "pass",
    });
    expect(useAuthStore.getState().isAuthenticated).toBe(true);
  });

  it("register calls API and sets auth state", async () => {
    vi.mocked(authApi.getMe).mockRejectedValueOnce(new Error("No session"));
    vi.mocked(authApi.register).mockResolvedValueOnce(mockTokenResponse);
    vi.mocked(authApi.getMe).mockResolvedValueOnce(mockUser);

    const { result } = renderHook(() => useAuth());

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    await result.current.register({
      email: "new@example.com",
      username: "new",
      password: "pass",
    });

    expect(authApi.register).toHaveBeenCalled();
    expect(useAuthStore.getState().isAuthenticated).toBe(true);
  });

  it("logout clears auth state", async () => {
    vi.mocked(authApi.getMe).mockRejectedValueOnce(new Error("No session"));
    vi.mocked(authApi.logout).mockResolvedValueOnce(undefined);

    useAuthStore.getState().setAuth(mockUser, "tok");

    const { result } = renderHook(() => useAuth());

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    await result.current.logout();

    expect(authApi.logout).toHaveBeenCalled();
    expect(useAuthStore.getState().isAuthenticated).toBe(false);
  });
});
