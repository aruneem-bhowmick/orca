/**
 * Zustand store for the application theme preference.
 *
 * Supports three modes: `"light"`, `"dark"`, and `"system"`. In system mode
 * the applied theme follows the OS preference detected via the
 * `prefers-color-scheme` media query. The selected preference is persisted to
 * `localStorage` so it survives page reloads.
 *
 * `applyTheme()` must be called once on application startup (and after each
 * toggle) to write the `dark` class to `document.documentElement`, which
 * activates Tailwind's dark-mode variant. `initTheme()` is a convenience
 * function that reads the stored preference and calls `applyTheme` immediately.
 *
 * @module store/theme
 */
import { create } from "zustand";

/** The three supported theme preferences. */
export type ThemeMode = "light" | "dark" | "system";

/** Shape of the theme store and its action methods. */
export interface ThemeState {
  /** The user-selected theme mode. */
  mode: ThemeMode;
  /**
   * Change the theme mode and immediately apply it to the DOM.
   *
   * Persists the new mode to localStorage under the key `"themeMode"`.
   *
   * @param mode - The new mode to activate.
   */
  setMode: (mode: ThemeMode) => void;
  /**
   * Toggle between light and dark modes.
   *
   * If the current mode is `"system"`, resolves the effective mode from the
   * OS preference before toggling.
   */
  toggle: () => void;
}

const STORAGE_KEY = "themeMode";

/**
 * Determine whether dark mode should be active for a given preference mode.
 *
 * In `"system"` mode, queries the `prefers-color-scheme` media API.
 *
 * @param mode - The configured theme mode.
 * @returns `true` when dark mode should be applied.
 */
export function resolveIsDark(mode: ThemeMode): boolean {
  if (mode === "dark") return true;
  if (mode === "light") return false;
  if (typeof window !== "undefined") {
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  return false;
}

/**
 * Write or remove the `dark` class on `document.documentElement` to
 * activate Tailwind's dark-mode variant.
 *
 * @param mode - The theme mode to apply.
 */
export function applyTheme(mode: ThemeMode): void {
  if (typeof document === "undefined") return;
  if (resolveIsDark(mode)) {
    document.documentElement.classList.add("dark");
  } else {
    document.documentElement.classList.remove("dark");
  }
}

/**
 * Read the persisted theme preference from localStorage and apply it to the
 * DOM. Call this once during application startup (before first render) to
 * prevent a flash of the wrong theme.
 *
 * @returns The resolved `ThemeMode` that was applied.
 */
export function initTheme(): ThemeMode {
  const stored = (
    typeof localStorage !== "undefined"
      ? localStorage.getItem(STORAGE_KEY)
      : null
  ) as ThemeMode | null;
  const mode: ThemeMode =
    stored === "light" || stored === "dark" || stored === "system"
      ? stored
      : "system";
  applyTheme(mode);
  return mode;
}

/**
 * Zustand store for the application theme.
 *
 * Reading `mode` returns the user-configured preference, not the effective
 * applied theme (which may differ when mode is `"system"`). Use
 * `resolveIsDark(mode)` to determine the effective applied theme.
 */
export const useThemeStore = create<ThemeState>((set, get) => ({
  mode: initTheme(),

  setMode: (mode) => {
    applyTheme(mode);
    if (typeof localStorage !== "undefined") {
      localStorage.setItem(STORAGE_KEY, mode);
    }
    set({ mode });
  },

  toggle: () => {
    const { mode, setMode } = get();
    const isDark = resolveIsDark(mode);
    setMode(isDark ? "light" : "dark");
  },
}));

if (typeof window !== "undefined" && typeof window.matchMedia === "function") {
  try {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handleMediaChange = () => {
      const currentMode = useThemeStore.getState().mode;
      if (currentMode === "system") {
        applyTheme("system");
      }
    };
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener("change", handleMediaChange);
    } else if ((mediaQuery as any).addListener) {
      (mediaQuery as any).addListener(handleMediaChange);
    }
  } catch (e) {
    // Handle environments with partially mocked matchMedia APIs gracefully.
  }
}
