import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { ToastContainer } from "@/components/ui/Toast";
import { useNotificationsStore } from "@/store/notifications";

function renderToasts() {
  return render(<ToastContainer />);
}

describe("ToastContainer", () => {
  beforeEach(() => {
    useNotificationsStore.getState().clearNotifications();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    useNotificationsStore.getState().clearNotifications();
  });

  it("renders nothing when there are no notifications", () => {
    const { container } = renderToasts();
    expect(container.firstChild).toBeNull();
  });

  it("renders a toast when a notification is added", () => {
    act(() => {
      useNotificationsStore.getState().addNotification("Test message", "info");
    });
    renderToasts();
    expect(screen.getByTestId("toast-container")).toBeInTheDocument();
    expect(screen.getByText("Test message")).toBeInTheDocument();
  });

  it("dismisses a toast when the close button is clicked", () => {
    act(() => {
      useNotificationsStore.getState().addNotification("Dismiss me", "info");
    });
    renderToasts();

    const { notifications } = useNotificationsStore.getState();
    const id = notifications[0].id;

    const closeBtn = screen.getByTestId(`toast-close-${id}`);
    fireEvent.click(closeBtn);

    expect(useNotificationsStore.getState().notifications).toHaveLength(0);
  });

  it("renders error level notifications with role=alert", () => {
    act(() => {
      useNotificationsStore.getState().addNotification("Something failed", "error");
    });
    renderToasts();
    const alerts = screen.getAllByRole("alert");
    expect(alerts.length).toBeGreaterThan(0);
    expect(alerts[0]).toHaveTextContent("Something failed");
  });

  it("auto-dismisses a toast after the timeout", () => {
    act(() => {
      useNotificationsStore.getState().addNotification("Auto-dismiss", "success");
    });
    renderToasts();

    expect(useNotificationsStore.getState().notifications).toHaveLength(1);

    act(() => {
      vi.advanceTimersByTime(4001);
    });

    expect(useNotificationsStore.getState().notifications).toHaveLength(0);
  });

  it("renders multiple notifications stacked", () => {
    act(() => {
      useNotificationsStore.getState().addNotification("First", "info");
      useNotificationsStore.getState().addNotification("Second", "warning");
    });
    renderToasts();
    expect(screen.getByText("First")).toBeInTheDocument();
    expect(screen.getByText("Second")).toBeInTheDocument();
  });
});
