import type { User, TokenResponse, HealthStatus } from "@/api/types";

export const mockUser: User = {
  user_id: "550e8400-e29b-41d4-a716-446655440000",
  email: "test@example.com",
  username: "testuser",
  role: "user",
  preferences: null,
};

export const mockTokenResponse: TokenResponse = {
  access_token: "mock-access-token-jwt",
  token_type: "bearer",
};

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
