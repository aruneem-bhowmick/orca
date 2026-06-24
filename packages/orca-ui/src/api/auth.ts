import axios from "axios";
import apiClient from "./client";
import { API_BASE_URL } from "@/lib/constants";
import type { LoginRequest, RegisterRequest, TokenResponse, User } from "./types";

export async function login(data: LoginRequest): Promise<TokenResponse> {
  const response = await apiClient.post<TokenResponse>("/auth/login", data);
  return response.data;
}

export async function register(data: RegisterRequest): Promise<TokenResponse> {
  const response = await apiClient.post<TokenResponse>("/auth/register", data);
  return response.data;
}

export async function refreshToken(): Promise<TokenResponse> {
  const response = await axios.post<TokenResponse>(
    `${API_BASE_URL}/auth/refresh`,
    {},
    { withCredentials: true },
  );
  return response.data;
}

export async function logout(): Promise<void> {
  await apiClient.post("/auth/logout");
}

export async function getMe(): Promise<User> {
  const response = await apiClient.get<User>("/auth/me");
  return response.data;
}

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
