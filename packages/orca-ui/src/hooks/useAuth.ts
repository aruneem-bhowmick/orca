import { useCallback, useEffect, useState } from "react";
import axios from "axios";
import { useAuthStore } from "@/store/auth";
import * as authApi from "@/api/auth";
import type { LoginRequest, RegisterRequest } from "@/api/types";

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

  const login = useCallback(
    async (data: LoginRequest) => {
      const tokenResponse = await authApi.login(data);
      const me = await authApi.getMe();
      setAuth(me, tokenResponse.access_token);
    },
    [setAuth],
  );

  const register = useCallback(
    async (data: RegisterRequest) => {
      const tokenResponse = await authApi.register(data);
      const me = await authApi.getMe();
      setAuth(me, tokenResponse.access_token);
    },
    [setAuth],
  );

  const logout = useCallback(async () => {
    try {
      await authApi.logout();
    } finally {
      clearAuth();
    }
  }, [clearAuth]);

  return {
    user,
    isAuthenticated,
    isLoading,
    login,
    register,
    logout,
  };
}
