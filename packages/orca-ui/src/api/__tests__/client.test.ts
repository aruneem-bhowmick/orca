import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import axios from "axios";
import apiClient from "@/api/client";
import { useAuthStore } from "@/store/auth";
import { mockUser } from "@/test/mocks/handlers";
import type { InternalAxiosRequestConfig } from "axios";

interface InterceptorHandler<T> {
  fulfilled: (value: T) => T | Promise<T>;
  rejected?: (error: unknown) => unknown;
}

vi.mock("axios", async () => {
  const actual = await vi.importActual<typeof import("axios")>("axios");
  return {
    ...actual,
    default: {
      ...actual.default,
      create: vi.fn(() => {
        const instance = actual.default.create();
        return instance;
      }),
      post: vi.fn(),
    },
  };
});

describe("API client interceptors", () => {
  beforeEach(() => {
    useAuthStore.getState().clearAuth();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  function getRequestHandlers() {
    return (apiClient.interceptors.request as unknown as { handlers: InterceptorHandler<InternalAxiosRequestConfig>[] }).handlers;
  }

  function getResponseHandlers() {
    return (apiClient.interceptors.response as unknown as { handlers: InterceptorHandler<unknown>[] }).handlers;
  }

  it("attaches Authorization header when token is present", async () => {
    useAuthStore.getState().setAuth(mockUser, "my-jwt-token");

    const config = {
      headers: {
        Authorization: "",
        set: function (key: string, value: string) {
          this[key as keyof typeof this] = value as never;
        },
      },
    };

    const handlers = getRequestHandlers();
    expect(handlers.length).toBeGreaterThan(0);

    const result = (await handlers[0].fulfilled(
      config as unknown as InternalAxiosRequestConfig,
    )) as InternalAxiosRequestConfig;
    expect(result.headers.Authorization).toBe("Bearer my-jwt-token");
  });

  it("does not attach Authorization header when no token", async () => {
    const config = {
      headers: {
        set: function () {},
      },
    };

    const handlers = getRequestHandlers();
    const result = (await handlers[0].fulfilled(
      config as unknown as InternalAxiosRequestConfig,
    )) as InternalAxiosRequestConfig;
    expect(result.headers.Authorization).toBeUndefined();
  });

  it("clears auth and redirects on refresh failure", async () => {
    useAuthStore.getState().setAuth(mockUser, "expired-token");

    const mockPost = vi.mocked(axios.post);
    mockPost.mockRejectedValueOnce(new Error("refresh failed"));

    const handlers = getResponseHandlers();
    expect(handlers.length).toBeGreaterThan(0);
    expect(handlers[0].rejected).toBeInstanceOf(Function);

    const error = {
      response: { status: 401 },
      config: { headers: { Authorization: "" }, _retry: false },
    };

    try {
      await handlers[0].rejected!(error);
    } catch {
      // Expected to reject
    }

    expect(useAuthStore.getState().isAuthenticated).toBe(false);
    expect(useAuthStore.getState().accessToken).toBeNull();
  });
});
