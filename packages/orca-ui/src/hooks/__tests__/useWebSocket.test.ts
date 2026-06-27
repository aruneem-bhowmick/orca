import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useAuthStore } from "@/store/auth";
import type { MetricUpdate } from "@/api/types";

// ---------------------------------------------------------------------------
// Mock WebSocket
// ---------------------------------------------------------------------------

/** Minimal mock WebSocket that captures handlers and exposes test helpers. */
class MockWebSocket {
  static OPEN = 1;
  static CLOSED = 3;

  readyState = MockWebSocket.OPEN;
  url: string;
  protocols?: string | string[];

  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  sentMessages: string[] = [];

  constructor(url: string, protocols?: string | string[]) {
    this.url = url;
    this.protocols = protocols;
    MockWebSocket.instances.push(this);
  }

  send(data: string) {
    this.sentMessages.push(data);
  }

  close(code?: number, reason?: string) {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.({
      code: code ?? 1000,
      reason: reason ?? "",
      wasClean: true,
    } as CloseEvent);
  }

  // Test helpers
  triggerOpen() {
    this.onopen?.(new Event("open"));
  }

  triggerMessage(data: unknown) {
    this.onmessage?.({
      data: JSON.stringify(data),
    } as MessageEvent);
  }

  triggerClose(code = 1001) {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.({ code, reason: "", wasClean: false } as CloseEvent);
  }

  triggerError() {
    this.onerror?.(new Event("error"));
  }

  static instances: MockWebSocket[] = [];
  static latest(): MockWebSocket {
    return MockWebSocket.instances[MockWebSocket.instances.length - 1];
  }
  static reset() {
    MockWebSocket.instances = [];
  }
}

vi.stubGlobal("WebSocket", MockWebSocket);

// ---------------------------------------------------------------------------
// Seed auth store with a token so the hook can build the URL
// ---------------------------------------------------------------------------

beforeEach(() => {
  MockWebSocket.reset();
  useAuthStore.getState().setAuth(
    { user_id: "u1", email: "t@test.com", username: "t", role: "user", preferences: null },
    "test-token",
  );
});

afterEach(() => {
  useAuthStore.getState().clearAuth();
  vi.useRealTimers();
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const METRIC: MetricUpdate = {
  epoch: 1,
  loss: 0.5,
  accuracy: 0.75,
  learning_rate: 0.001,
  timestamp: "2024-01-01T00:00:00Z",
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("useWebSocket", () => {
  it("does not open a connection when experimentId is empty", () => {
    renderHook(() => useWebSocket(""));
    expect(MockWebSocket.instances).toHaveLength(0);
  });

  it("opens a connection with the correct URL when given an experimentId", () => {
    renderHook(() => useWebSocket("exp-123"));
    expect(MockWebSocket.instances).toHaveLength(1);
    expect(MockWebSocket.latest().url).toContain("/api/v1/orcalab/ws/experiments/exp-123/live");
    expect(MockWebSocket.latest().url).not.toContain("token=");
    expect(MockWebSocket.latest().protocols).toBe("test-token");
  });

  it("starts with isConnected false", () => {
    const { result } = renderHook(() => useWebSocket("exp-1"));
    expect(result.current.isConnected).toBe(false);
  });

  it("sets isConnected true when the socket opens", async () => {
    const { result } = renderHook(() => useWebSocket("exp-1"));
    act(() => {
      MockWebSocket.latest().triggerOpen();
    });
    await waitFor(() => expect(result.current.isConnected).toBe(true));
  });

  it("starts with an empty messages array", () => {
    const { result } = renderHook(() => useWebSocket("exp-1"));
    expect(result.current.messages).toHaveLength(0);
  });

  it("appends a MetricUpdate to messages when a message arrives", async () => {
    const { result } = renderHook(() => useWebSocket("exp-1"));
    act(() => {
      MockWebSocket.latest().triggerOpen();
      MockWebSocket.latest().triggerMessage(METRIC);
    });
    await waitFor(() => expect(result.current.messages).toHaveLength(1));
    expect(result.current.messages[0]).toEqual(METRIC);
  });

  it("accumulates multiple messages in order", async () => {
    const { result } = renderHook(() => useWebSocket("exp-1"));
    const second: MetricUpdate = { ...METRIC, epoch: 2, loss: 0.4 };
    act(() => {
      MockWebSocket.latest().triggerOpen();
      MockWebSocket.latest().triggerMessage(METRIC);
      MockWebSocket.latest().triggerMessage(second);
    });
    await waitFor(() => expect(result.current.messages).toHaveLength(2));
    expect(result.current.messages[0].epoch).toBe(1);
    expect(result.current.messages[1].epoch).toBe(2);
  });

  it("ignores malformed (non-JSON) messages", async () => {
    const { result } = renderHook(() => useWebSocket("exp-1"));
    act(() => {
      MockWebSocket.latest().triggerOpen();
      MockWebSocket.latest().onmessage?.({ data: "not json" } as MessageEvent);
    });
    await waitFor(() => {
      expect(result.current.messages).toHaveLength(0);
    });
  });

  it("ignores valid JSON messages with invalid MetricUpdate fields or types", async () => {
    const { result } = renderHook(() => useWebSocket("exp-1"));
    act(() => {
      MockWebSocket.latest().triggerOpen();
      // Missing epoch
      MockWebSocket.latest().triggerMessage({ loss: 0.5, timestamp: "2024-01-01T00:00:00Z" });
      // Wrong type for epoch (string instead of number)
      MockWebSocket.latest().triggerMessage({ epoch: "1", loss: 0.5, timestamp: "2024-01-01T00:00:00Z" });
      // Wrong type for accuracy
      MockWebSocket.latest().triggerMessage({ epoch: 1, loss: 0.5, accuracy: "high", timestamp: "2024-01-01T00:00:00Z" });
    });
    await new Promise((r) => setTimeout(r, 50));
    expect(result.current.messages).toHaveLength(0);
  });

  it("sets isConnected false on socket close", async () => {
    const { result } = renderHook(() => useWebSocket("exp-1"));
    act(() => {
      MockWebSocket.latest().triggerOpen();
    });
    await waitFor(() => expect(result.current.isConnected).toBe(true));
    act(() => {
      MockWebSocket.latest().triggerClose(1000);
    });
    await waitFor(() => expect(result.current.isConnected).toBe(false));
  });

  it("sends JSON-serialised data via send()", async () => {
    const { result } = renderHook(() => useWebSocket("exp-1"));
    act(() => {
      MockWebSocket.latest().triggerOpen();
    });
    await waitFor(() => expect(result.current.isConnected).toBe(true));
    act(() => {
      result.current.send({ action: "pause" });
    });
    expect(MockWebSocket.latest().sentMessages).toHaveLength(1);
    expect(JSON.parse(MockWebSocket.latest().sentMessages[0])).toEqual({
      action: "pause",
    });
  });

  it("does not send when the socket is not open", () => {
    const { result } = renderHook(() => useWebSocket("exp-1"));
    // Socket exists but has not received triggerOpen yet
    MockWebSocket.latest().readyState = MockWebSocket.CLOSED;
    act(() => {
      result.current.send({ action: "pause" });
    });
    expect(MockWebSocket.latest().sentMessages).toHaveLength(0);
  });

  it("close() sets isConnected false permanently", async () => {
    const { result } = renderHook(() => useWebSocket("exp-1"));
    act(() => {
      MockWebSocket.latest().triggerOpen();
    });
    await waitFor(() => expect(result.current.isConnected).toBe(true));
    act(() => {
      result.current.close();
    });
    // close() is synchronous; isConnected is set to false in the same tick
    expect(result.current.isConnected).toBe(false);
  });

  it("schedules a reconnect with backoff on unexpected close", async () => {
    vi.useFakeTimers();
    renderHook(() => useWebSocket("exp-1"));
    act(() => {
      MockWebSocket.latest().triggerOpen();
      MockWebSocket.latest().triggerClose(1006); // abnormal
    });
    expect(MockWebSocket.instances).toHaveLength(1); // not yet reconnected
    act(() => {
      vi.advanceTimersByTime(1_100); // past the 1 s initial backoff
    });
    expect(MockWebSocket.instances).toHaveLength(2); // new socket opened
  });

  it("does not reconnect after close()", async () => {
    vi.useFakeTimers();
    const { result } = renderHook(() => useWebSocket("exp-1"));
    act(() => {
      MockWebSocket.latest().triggerOpen();
    });
    act(() => {
      result.current.close();
    });
    act(() => {
      vi.advanceTimersByTime(5_000);
    });
    expect(MockWebSocket.instances).toHaveLength(1); // no additional socket
  });

  it("does not reconnect after auth-rejection close (code 4001)", async () => {
    vi.useFakeTimers();
    renderHook(() => useWebSocket("exp-1"));
    act(() => {
      MockWebSocket.latest().triggerOpen();
      MockWebSocket.latest().triggerClose(4001);
    });
    act(() => {
      vi.advanceTimersByTime(5_000);
    });
    expect(MockWebSocket.instances).toHaveLength(1);
  });

  it("clears messages and reconnects when experimentId changes", async () => {
    const { result, rerender } = renderHook(
      ({ id }: { id: string }) => useWebSocket(id),
      { initialProps: { id: "exp-A" } },
    );
    act(() => {
      MockWebSocket.latest().triggerOpen();
      MockWebSocket.latest().triggerMessage(METRIC);
    });
    await waitFor(() => expect(result.current.messages).toHaveLength(1));

    rerender({ id: "exp-B" });

    await waitFor(() => expect(result.current.messages).toHaveLength(0));
    expect(MockWebSocket.instances).toHaveLength(2);
    expect(MockWebSocket.instances[1].url).toContain("exp-B");
  });

  it("closes the socket on component unmount", () => {
    const { unmount } = renderHook(() => useWebSocket("exp-1"));
    unmount();
    expect(MockWebSocket.latest().readyState).toBe(MockWebSocket.CLOSED);
  });
});
