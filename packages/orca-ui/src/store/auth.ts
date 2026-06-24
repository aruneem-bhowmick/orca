import { create } from "zustand";
import type { User } from "@/api/types";

export interface AuthState {
  user: User | null;
  accessToken: string | null;
  isAuthenticated: boolean;
  setAuth: (user: User, accessToken: string) => void;
  setToken: (accessToken: string) => void;
  setUser: (user: User) => void;
  clearAuth: () => void;
}

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
