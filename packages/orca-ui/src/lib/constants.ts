/**
 * Application-wide constants.
 *
 * @module lib/constants
 */

/**
 * Base URL for all BFF API requests. Reads from the `VITE_API_BASE_URL`
 * environment variable at build time, falling back to `"/api/v1"` which
 * is handled by Vite's dev proxy or the production nginx config.
 */
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api/v1";

/**
 * All named route paths in the application.
 *
 * Public routes: HOME, LOGIN, REGISTER, OAUTH_CALLBACK.
 * Protected routes (require authentication): all others.
 */
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
