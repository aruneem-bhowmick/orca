export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api/v1";

export const ROUTES = {
  HOME: "/",
  LOGIN: "/login",
  REGISTER: "/register",
  OAUTH_CALLBACK: "/oauth/callback",
  DASHBOARD: "/dashboard",
  TASKS: "/tasks",
  EXPERIMENTS: "/experiments",
  SWEEPS: "/sweeps",
  TRANSFERS: "/transfers",
  HISTORY: "/history",
  BOOKMARKS: "/bookmarks",
  SETTINGS: "/settings",
} as const;
