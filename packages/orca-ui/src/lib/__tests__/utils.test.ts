import { describe, it, expect } from "vitest";
import { formatElapsed } from "@/lib/utils";

describe("formatElapsed", () => {
  it("formats elapsed duration between valid start and end date strings", () => {
    const start = "2024-03-15T10:00:00Z";
    const end = "2024-03-15T10:15:30Z";
    expect(formatElapsed(start, end)).toBe("15m 30s");
  });

  it("formats elapsed duration when duration is over an hour", () => {
    const start = "2024-03-15T10:00:00Z";
    const end = "2024-03-15T12:15:30Z";
    expect(formatElapsed(start, end)).toBe("2h 15m");
  });

  it("formats elapsed duration in seconds when under a minute", () => {
    const start = "2024-03-15T10:00:00Z";
    const end = "2024-03-15T10:00:08Z";
    expect(formatElapsed(start, end)).toBe("8s");
  });

  it("returns fallback for invalid startDateString", () => {
    expect(formatElapsed("invalid-date", "2024-03-15T10:15:30Z")).toBe("—");
  });

  it("returns fallback for invalid endDateString", () => {
    expect(formatElapsed("2024-03-15T10:00:00Z", "invalid-date")).toBe("—");
  });
});
