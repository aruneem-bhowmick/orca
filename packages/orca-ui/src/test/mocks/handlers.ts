/**
 * Mock data fixtures for orca-ui tests.
 *
 * Provides typed mock objects matching the BFF's response schemas.
 * Used across all test suites to ensure consistent test data.
 *
 * @module test/mocks/handlers
 */
import type { User, TokenResponse, HealthStatus } from "@/api/types";

/** Mock authenticated user profile. */
export const mockUser: User = {
  user_id: "550e8400-e29b-41d4-a716-446655440000",
  email: "test@example.com",
  username: "testuser",
  role: "user",
  preferences: null,
};

/** Mock JWT token response from login/register/refresh endpoints. */
export const mockTokenResponse: TokenResponse = {
  access_token: "mock-access-token-jwt",
  token_type: "bearer",
};

/** Mock health check response with all services healthy. */
export const mockHealthStatus: HealthStatus = {
  status: "healthy",
  services: {
    postgres: true,
    redis: true,
    orcamind: true,
    orcalab: true,
    orcanet: true,
  },
};
