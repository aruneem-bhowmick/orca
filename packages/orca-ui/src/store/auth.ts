import { create } from "zustand";
import type { User } from "@/api/types";

export interface AuthState {
  user: User | null;
  accessToken: string | null;
  isAuthenticated: boolean;
  setAuth: (user: User, accessToken: string) => void;
  clearAuth: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  accessToken: null,
  isAuthenticated: false,
  setAuth: (user: User, accessToken: string) =>
    set({ user, accessToken, isAuthenticated: true }),
  clearAuth: () => set({ user: null, accessToken: null, isAuthenticated: false }),
}));
