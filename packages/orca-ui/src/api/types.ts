export interface User {
  user_id: string;
  email: string;
  username: string;
  role: string;
  preferences: Record<string, unknown> | null;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface RegisterRequest {
  email: string;
  username: string;
  password: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface ProfileUpdate {
  username?: string;
  preferences?: Record<string, unknown>;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  per_page: number;
  pages: number;
}

export interface ActivityLogEntry {
  id: string;
  user_id: string;
  action: string;
  resource_type: string | null;
  resource_id: string | null;
  service: string | null;
  details: Record<string, unknown> | null;
  created_at: string;
}

export interface Bookmark {
  id: string;
  user_id: string;
  resource_type: string;
  resource_id: string;
  note: string | null;
  created_at: string;
}

export interface BookmarkCreateRequest {
  resource_type: string;
  resource_id: string;
  note?: string;
}

export interface HealthStatus {
  status: "healthy" | "degraded";
  services: {
    postgres: boolean;
    redis: boolean;
    orcamind: boolean;
    orcalab: boolean;
    orcanet: boolean;
  };
}
