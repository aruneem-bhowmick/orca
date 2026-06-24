import { describe, it, expect, vi, beforeEach } from "vitest";
import apiClient from "@/api/client";
import { login, register, logout, getMe } from "@/api/auth";
import { mockUser, mockTokenResponse } from "@/test/mocks/handlers";

vi.mock("@/api/client", () => ({
  default: {
    post: vi.fn(),
    get: vi.fn(),
    interceptors: {
      request: { use: vi.fn(), handlers: [] },
      response: { use: vi.fn(), handlers: [] },
    },
  },
}));

vi.mock("axios", () => ({
  default: {
    post: vi.fn(),
  },
}));

describe("auth API functions", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("login", () => {
    it("sends POST /auth/login with credentials", async () => {
      vi.mocked(apiClient.post).mockResolvedValueOnce({ data: mockTokenResponse });

      const result = await login({ email: "test@example.com", password: "password123" });

      expect(apiClient.post).toHaveBeenCalledWith("/auth/login", {
        email: "test@example.com",
        password: "password123",
      });
      expect(result).toEqual(mockTokenResponse);
    });

    it("propagates 401 errors from the BFF", async () => {
      const error = {
        response: { status: 401, data: { detail: "Invalid credentials" } },
      };
      vi.mocked(apiClient.post).mockRejectedValueOnce(error);

      await expect(
        login({ email: "wrong@example.com", password: "wrong" }),
      ).rejects.toMatchObject({
        response: { status: 401, data: { detail: "Invalid credentials" } },
      });
    });
  });

  describe("register", () => {
    it("sends POST /auth/register with user data", async () => {
      vi.mocked(apiClient.post).mockResolvedValueOnce({ data: mockTokenResponse });

      const result = await register({
        email: "new@example.com",
        username: "newuser",
        password: "password123",
      });

      expect(apiClient.post).toHaveBeenCalledWith("/auth/register", {
        email: "new@example.com",
        username: "newuser",
        password: "password123",
      });
      expect(result).toEqual(mockTokenResponse);
    });

    it("propagates 409 errors on duplicate email", async () => {
      const error = {
        response: { status: 409, data: { detail: "Email already registered" } },
      };
      vi.mocked(apiClient.post).mockRejectedValueOnce(error);

      await expect(
        register({
          email: "existing@example.com",
          username: "existing",
          password: "password123",
        }),
      ).rejects.toMatchObject({
        response: { status: 409, data: { detail: "Email already registered" } },
      });
    });
  });

  describe("logout", () => {
    it("sends POST /auth/logout", async () => {
      vi.mocked(apiClient.post).mockResolvedValueOnce({ data: undefined });

      await logout();

      expect(apiClient.post).toHaveBeenCalledWith("/auth/logout");
    });
  });

  describe("getMe", () => {
    it("sends GET /auth/me and returns user", async () => {
      vi.mocked(apiClient.get).mockResolvedValueOnce({ data: mockUser });

      const result = await getMe();

      expect(apiClient.get).toHaveBeenCalledWith("/auth/me");
      expect(result).toEqual(mockUser);
    });
  });
});
