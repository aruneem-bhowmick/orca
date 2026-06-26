import { describe, it, expect, beforeEach } from "vitest";
import { useAuthStore } from "@/store/auth";
import { mockUser } from "@/test/mocks/handlers";

describe("useAuthStore", () => {
  beforeEach(() => {
    useAuthStore.getState().clearAuth();
  });

  it("starts with unauthenticated state", () => {
    const state = useAuthStore.getState();
    expect(state.user).toBeNull();
    expect(state.accessToken).toBeNull();
    expect(state.isAuthenticated).toBe(false);
  });

  it("setAuth updates user, token, and isAuthenticated", () => {
    useAuthStore.getState().setAuth(mockUser, "test-token");
    const state = useAuthStore.getState();
    expect(state.user).toEqual(mockUser);
    expect(state.accessToken).toBe("test-token");
    expect(state.isAuthenticated).toBe(true);
  });

  it("clearAuth resets to unauthenticated state", () => {
    useAuthStore.getState().setAuth(mockUser, "test-token");
    useAuthStore.getState().clearAuth();
    const state = useAuthStore.getState();
    expect(state.user).toBeNull();
    expect(state.accessToken).toBeNull();
    expect(state.isAuthenticated).toBe(false);
  });

  it("isAuthenticated is derived from setAuth/clearAuth", () => {
    expect(useAuthStore.getState().isAuthenticated).toBe(false);
    useAuthStore.getState().setAuth(mockUser, "tok");
    expect(useAuthStore.getState().isAuthenticated).toBe(true);
    useAuthStore.getState().clearAuth();
    expect(useAuthStore.getState().isAuthenticated).toBe(false);
  });

  it("setToken updates only the access token", () => {
    useAuthStore.getState().setAuth(mockUser, "original-token");
    useAuthStore.getState().setToken("refreshed-token");

    const state = useAuthStore.getState();
    expect(state.accessToken).toBe("refreshed-token");
    expect(state.user).toEqual(mockUser);
    expect(state.isAuthenticated).toBe(true);
  });

  it("setUser updates user and marks as authenticated", () => {
    useAuthStore.getState().setUser(mockUser);

    const state = useAuthStore.getState();
    expect(state.user).toEqual(mockUser);
    expect(state.isAuthenticated).toBe(true);
    expect(state.accessToken).toBeNull();
  });
});
