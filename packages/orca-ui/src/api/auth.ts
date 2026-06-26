/**
 * Authentication API functions for communicating with the Orca Web BFF
 * auth endpoints. All functions return typed responses matching the
 * BFF's Pydantic schemas.
 *
 * @module api/auth
 */
import axios from "axios";
import apiClient from "./client";
import { API_BASE_URL } from "@/lib/constants";
import type { LoginRequest, RegisterRequest, TokenResponse, User } from "./types";

/**
 * Authenticate with email and password.
 *
 * @param data - Login credentials (email, password).
 * @returns A TokenResponse containing the JWT access token.
 * @throws AxiosError with status 401 for invalid credentials.
 */
export async function login(data: LoginRequest): Promise<TokenResponse> {
  const response = await apiClient.post<TokenResponse>("/auth/login", data);
  return response.data;
}

/**
 * Register a new user account.
 *
 * @param data - Registration data (email, username, password).
 * @returns A TokenResponse containing the JWT access token.
 * @throws AxiosError with status 409 for duplicate email or username.
 */
export async function register(data: RegisterRequest): Promise<TokenResponse> {
  const response = await apiClient.post<TokenResponse>("/auth/register", data);
  return response.data;
}

/**
 * Refresh the access token using the httponly refresh cookie.
 *
 * Uses a direct axios call (not the apiClient) to avoid triggering
 * the response interceptor's own refresh logic, which would cause
 * infinite recursion on 401.
 *
 * @returns A TokenResponse with the new access token.
 * @throws AxiosError with status 401 if the refresh token is invalid.
 */
export async function refreshToken(): Promise<TokenResponse> {
  const response = await axios.post<TokenResponse>(
    `${API_BASE_URL}/auth/refresh`,
    {},
    { withCredentials: true, timeout: 10_000 },
  );
  return response.data;
}

/**
 * Log out the current user by revoking the refresh token server-side.
 * The BFF clears the refresh cookie on a successful logout.
 */
export async function logout(): Promise<void> {
  await apiClient.post("/auth/logout");
}

/**
 * Fetch the currently authenticated user's profile.
 *
 * @returns The authenticated user's profile data.
 * @throws AxiosError with status 401 if no valid session exists.
 */
export async function getMe(): Promise<User> {
  const response = await apiClient.get<User>("/auth/me");
  return response.data;
}

/**
 * Exchange an OAuth authorization code for an access token.
 *
 * Called by the OAuthCallback page after the OAuth provider redirects
 * back with query parameters. Uses a direct axios call with
 * `withCredentials: true` so the BFF can set the refresh cookie.
 *
 * @param provider - OAuth provider name ("google" or "github").
 * @param params - URL search parameters from the OAuth redirect.
 * @returns A TokenResponse containing the JWT access token.
 */
export async function exchangeOAuthCode(
  provider: string,
  params: URLSearchParams,
): Promise<TokenResponse> {
  const response = await axios.get<TokenResponse>(
    `${API_BASE_URL}/auth/oauth/${provider}/callback`,
    { params: Object.fromEntries(params), withCredentials: true },
  );
  return response.data;
}
