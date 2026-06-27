import { useCallback, useEffect, useRef, useState } from "react";
import { useAuthStore } from "@/store/auth";
import type { MetricUpdate } from "@/api/types";

/** Maximum number of reconnection attempts before giving up. */
const MAX_RETRIES = 3;

/** Initial backoff delay in milliseconds, doubled on each retry. */
const INITIAL_BACKOFF_MS = 1_000;

/**
 * Derive the WebSocket endpoint URL for a live experiment stream.
 *
 * The base is taken from `VITE_WS_BASE_URL` (e.g. `ws://localhost:8003`);
 * if that variable is absent the scheme is inferred from the current page
 * protocol so the hook works in both dev (http) and production (https).
 *
 * @param experimentId - The experiment to stream metrics for.
 * @param token - The JWT access token appended as a query parameter.
 * @returns Fully-qualified `ws://` or `wss://` URL.
 */
function buildWsUrl(experimentId: string): string {
  const base =
    import.meta.env.VITE_WS_BASE_URL ||
    (window.location.protocol === "https:" ? "wss:" : "ws:") +
      `//${window.location.host}`;
  return `${base}/api/v1/orcalab/ws/experiments/${experimentId}/live`;
}

/**
 * Return value of the `useWebSocket` hook.
 */
export interface UseWebSocketResult {
  /** Ordered list of metric updates received since the connection opened. */
  messages: MetricUpdate[];
  /** Whether the WebSocket connection is currently open. */
  isConnected: boolean;
  /**
   * Send a raw JSON-serialisable payload to the BFF over the open socket.
   * No-ops if the connection is not open.
   *
   * @param data - Any JSON-serialisable object.
   */
  send: (data: unknown) => void;
  /** Permanently close the WebSocket and suppress any further reconnection. */
  close: () => void;
}

/**
 * Manages a WebSocket connection to the live experiment metrics endpoint.
 *
 * Opens a connection at
 * `WS /api/v1/orcalab/ws/experiments/{experimentId}/live?token={accessToken}`
 * and parses each received frame as a `MetricUpdate`.
 *
 * Reconnection is attempted automatically on unexpected closure with
 * exponential backoff (1 s → 2 s → 4 s). After `MAX_RETRIES` (3)
 * failed attempts the hook stops reconnecting and `isConnected` stays
 * `false`. Calling `close()` permanently suppresses reconnection.
 *
 * The hook is safe to call when `experimentId` is empty – no connection
 * is opened in that case.
 *
 * @param experimentId - OrcaLab experiment ID whose metrics to stream.
 *   Pass an empty string to keep the hook idle.
 * @returns Connection state and control handles.
 */
export function useWebSocket(experimentId: string): UseWebSocketResult {
  const [messages, setMessages] = useState<MetricUpdate[]>([]);
  const [isConnected, setIsConnected] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const retriesRef = useRef(0);
  const permanentlyClosedRef = useRef(false);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    if (permanentlyClosedRef.current || !experimentId) return;

    const token = useAuthStore.getState().accessToken;
    if (!token) return;

    const url = buildWsUrl(experimentId);
    const ws = new WebSocket(url, token);
    wsRef.current = ws;

    ws.onopen = () => {
      retriesRef.current = 0;
      setIsConnected(true);
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const update = JSON.parse(event.data as string) as MetricUpdate;
        setMessages((prev) => [...prev, update]);
      } catch {
        // Ignore malformed frames
      }
    };

    ws.onclose = (event: CloseEvent) => {
      setIsConnected(false);
      wsRef.current = null;

      if (permanentlyClosedRef.current) return;

      // Normal closure codes (1000 = normal, 4001 = auth rejection from BFF)
      if (event.code === 1000 || event.code === 4001) return;

      if (retriesRef.current < MAX_RETRIES) {
        const delay = INITIAL_BACKOFF_MS * 2 ** retriesRef.current;
        retriesRef.current += 1;
        reconnectTimerRef.current = setTimeout(connect, delay);
      }
    };

    ws.onerror = () => {
      // onclose fires immediately after, which handles retry logic
    };
  }, [experimentId]);

  useEffect(() => {
    if (!experimentId) return;

    permanentlyClosedRef.current = false;
    retriesRef.current = 0;
    setMessages([]);
    connect();

    return () => {
      permanentlyClosedRef.current = true;
      clearReconnectTimer();
      if (wsRef.current) {
        wsRef.current.close(1000, "component unmounted");
        wsRef.current = null;
      }
    };
  }, [experimentId, connect, clearReconnectTimer]);

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  const close = useCallback(() => {
    permanentlyClosedRef.current = true;
    clearReconnectTimer();
    if (wsRef.current) {
      wsRef.current.close(1000, "user closed");
      wsRef.current = null;
    }
    setIsConnected(false);
  }, [clearReconnectTimer]);

  return { messages, isConnected, send, close };
}
