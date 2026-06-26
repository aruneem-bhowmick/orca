/**
 * Configured Axios instance for all BFF API communication.
 *
 * Attaches the JWT access token from the Zustand auth store to every
 * outgoing request. On 401 responses, automatically attempts a
 * cookie-based token refresh and retries the original request. If the
 * refresh fails, clears the auth store and redirects to the login page.
 * Concurrent 401 responses are queued so that only a single refresh
 * request is made at a time.
 *
 * @module api/client
 */
import axios from "axios";
import type { AxiosError, InternalAxiosRequestConfig } from "axios";
import { useAuthStore } from "@/store/auth";
import { API_BASE_URL, ROUTES } from "@/lib/constants";

/** Axios instance preconfigured with the BFF base URL and JSON content type. */
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

apiClient.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const token = useAuthStore.getState().accessToken;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

/** Whether a token refresh request is currently in flight. */
let isRefreshing = false;

/** Queue of requests waiting for the current refresh to complete. */
let failedQueue: Array<{
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
}> = [];

/**
 * Resolve or reject all queued requests after a refresh attempt.
 *
 * @param error - The refresh error (null on success).
 * @param token - The new access token (null on failure).
 */
function processQueue(error: unknown, token: string | null = null) {
  failedQueue.forEach((prom) => {
    if (error) {
      prom.reject(error);
    } else {
      prom.resolve(token);
    }
  });
  failedQueue = [];
}

apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & {
      _retry?: boolean;
    };

    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        }).then((token) => {
          originalRequest.headers.Authorization = `Bearer ${token}`;
          return apiClient(originalRequest);
        });
      }

      originalRequest._retry = true;
      isRefreshing = true;

      try {
        const response = await axios.post(
          `${API_BASE_URL}/auth/refresh`,
          {},
          { withCredentials: true, timeout: 10_000 },
        );
        const { access_token } = response.data;
        const store = useAuthStore.getState();
        const user = store.user;
        if (user) {
          store.setAuth(user, access_token);
        } else {
          store.setToken(access_token);
        }
        originalRequest.headers.Authorization = `Bearer ${access_token}`;
        processQueue(null, access_token);
        return apiClient(originalRequest);
      } catch (refreshError) {
        processQueue(refreshError, null);
        useAuthStore.getState().clearAuth();
        window.location.href = ROUTES.LOGIN;
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }

    return Promise.reject(error);
  },
);

export default apiClient;
