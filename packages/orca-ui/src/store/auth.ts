/**
 * Zustand store for authentication state management.
 *
 * Holds the current user profile, JWT access token, and a derived
 * `isAuthenticated` flag. The store is synchronous and does not
 * persist state — session restoration is handled by the `useAuth`
 * hook calling `GET /auth/me` on mount.
 *
 * @module store/auth
 */
import { create } from "zustand";
import type { User } from "@/api/types";

/** Shape of the authentication state and its action methods. */
export interface AuthState {
  /** The authenticated user's profile, or null when logged out. */
  user: User | null;
  /** The current JWT access token, or null when logged out. */
  accessToken: string | null;
  /** Whether a user is currently authenticated. */
  isAuthenticated: boolean;
  /** Set both the user profile and access token, marking the user as authenticated. */
  setAuth: (user: User, accessToken: string) => void;
  /** Update only the access token (e.g. after a token refresh). */
  setToken: (accessToken: string) => void;
  /** Update only the user profile, marking the user as authenticated. */
  setUser: (user: User) => void;
  /** Clear all auth state, returning to the logged-out state. */
  clearAuth: () => void;
}

/**
 * Zustand store instance for authentication.
 *
 * Can be used directly outside React via `useAuthStore.getState()`
 * (e.g. in the Axios interceptor) or as a React hook inside components.
 */
export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  accessToken: null,
  isAuthenticated: false,
  setAuth: (user: User, accessToken: string) =>
    set({ user, accessToken, isAuthenticated: true }),
  setToken: (accessToken: string) =>
    set({ accessToken }),
  setUser: (user: User) =>
    set({ user, isAuthenticated: true }),
  clearAuth: () => set({ user: null, accessToken: null, isAuthenticated: false }),
}));
