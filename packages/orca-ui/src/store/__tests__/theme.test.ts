import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { resolveIsDark, applyTheme, initTheme, useThemeStore } from "@/store/theme";

// jsdom provides document and localStorage, so we can test DOM side-effects.

describe("resolveIsDark", () => {
  it("returns true for 'dark' mode", () => {
    expect(resolveIsDark("dark")).toBe(true);
  });

  it("returns false for 'light' mode", () => {
    expect(resolveIsDark("light")).toBe(false);
  });

  it("defers to matchMedia for 'system' mode — dark OS", () => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: vi.fn().mockReturnValue({ matches: true }),
    });
    expect(resolveIsDark("system")).toBe(true);
  });

  it("defers to matchMedia for 'system' mode — light OS", () => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: vi.fn().mockReturnValue({ matches: false }),
    });
    expect(resolveIsDark("system")).toBe(false);
  });
});

describe("applyTheme", () => {
  beforeEach(() => {
    document.documentElement.classList.remove("dark");
  });

  it("adds 'dark' class for dark mode", () => {
    applyTheme("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
  });

  it("removes 'dark' class for light mode", () => {
    document.documentElement.classList.add("dark");
    applyTheme("light");
    expect(document.documentElement.classList.contains("dark")).toBe(false);
  });
});

describe("initTheme", () => {
  afterEach(() => {
    localStorage.clear();
    document.documentElement.classList.remove("dark");
  });

  it("returns 'system' when localStorage has no entry", () => {
    localStorage.removeItem("themeMode");
    const mode = initTheme();
    expect(mode).toBe("system");
  });

  it("returns the stored mode for valid values", () => {
    localStorage.setItem("themeMode", "dark");
    const mode = initTheme();
    expect(mode).toBe("dark");
  });

  it("falls back to 'system' for an invalid stored value", () => {
    localStorage.setItem("themeMode", "invalid");
    const mode = initTheme();
    expect(mode).toBe("system");
  });
});

describe("useThemeStore", () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.classList.remove("dark");
    // Reset store to 'light' for predictable state in each test.
    useThemeStore.getState().setMode("light");
  });

  it("setMode updates the stored mode", () => {
    useThemeStore.getState().setMode("dark");
    expect(useThemeStore.getState().mode).toBe("dark");
  });

  it("setMode persists to localStorage", () => {
    useThemeStore.getState().setMode("dark");
    expect(localStorage.getItem("themeMode")).toBe("dark");
  });

  it("setMode applies dark class for dark mode", () => {
    useThemeStore.getState().setMode("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
  });

  it("setMode removes dark class for light mode", () => {
    document.documentElement.classList.add("dark");
    useThemeStore.getState().setMode("light");
    expect(document.documentElement.classList.contains("dark")).toBe(false);
  });

  it("toggle switches from light to dark", () => {
    useThemeStore.getState().setMode("light");
    useThemeStore.getState().toggle();
    expect(useThemeStore.getState().mode).toBe("dark");
  });

  it("toggle switches from dark to light", () => {
    useThemeStore.getState().setMode("dark");
    useThemeStore.getState().toggle();
    expect(useThemeStore.getState().mode).toBe("light");
  });
});
