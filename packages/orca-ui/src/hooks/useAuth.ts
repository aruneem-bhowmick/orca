import { useCallback, useEffect, useState } from "react";
import axios from "axios";
import { useAuthStore } from "@/store/auth";
import * as authApi from "@/api/auth";
import type { LoginRequest, RegisterRequest } from "@/api/types";

/**
 * Authentication hook that wraps the Zustand auth store with
 * convenience methods for login, registration, logout, and token
 * refresh.
 *
 * On mount, attempts to restore the current session by calling
 * `GET /auth/me`. If the call succeeds and a stored token exists,
 * the auth state is updated. If the call returns 401 or 403, the
 * auth state is cleared. The `isLoading` flag is `true` during
 * this initial check so the UI can show a loading indicator.
 *
 * @returns Auth state (`user`, `isAuthenticated`, `isLoading`) and
 *   action methods (`login`, `register`, `logout`, `refreshToken`).
 */
export function useAuth() {
  const { user, isAuthenticated, setAuth, clearAuth } = useAuthStore();
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function restoreSession() {
      try {
        const me = await authApi.getMe();
        if (!cancelled) {
          const token = useAuthStore.getState().accessToken;
          if (token) {
            setAuth(me, token);
          }
        }
      } catch (err) {
        if (!cancelled) {
          const status = axios.isAxiosError(err) ? err.response?.status : undefined;
          if (status === 401 || status === 403) {
            clearAuth();
          }
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    restoreSession();

    return () => {
      cancelled = true;
    };
  }, [setAuth, clearAuth]);

  /**
   * Authenticate with email and password. On success, fetches the
   * user profile and stores the access token in the auth store.
   */
  const login = useCallback(
    async (data: LoginRequest) => {
      const tokenResponse = await authApi.login(data);
      const me = await authApi.getMe();
      setAuth(me, tokenResponse.access_token);
    },
    [setAuth],
  );

  /**
   * Register a new account. On success, fetches the user profile
   * and stores the access token in the auth store.
   */
  const register = useCallback(
    async (data: RegisterRequest) => {
      const tokenResponse = await authApi.register(data);
      const me = await authApi.getMe();
      setAuth(me, tokenResponse.access_token);
    },
    [setAuth],
  );

  /**
   * Log out the current user. Calls the logout API endpoint and
   * clears the auth store regardless of whether the API call
   * succeeds (the store is always cleared in the `finally` block).
   */
  const logout = useCallback(async () => {
    try {
      await authApi.logout();
    } finally {
      clearAuth();
    }
  }, [clearAuth]);

  /**
   * Manually refresh the access token using the httponly refresh
   * cookie. Updates the token in the auth store on success.
   */
  const refreshToken = useCallback(async () => {
    const tokenResponse = await authApi.refreshToken();
    useAuthStore.getState().setToken(tokenResponse.access_token);
  }, []);

  return {
    user,
    isAuthenticated,
    isLoading,
    login,
    register,
    logout,
    refreshToken,
  };
}
