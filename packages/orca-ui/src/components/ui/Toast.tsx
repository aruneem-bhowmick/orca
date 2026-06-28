/**
 * Toast notification component connected to the notifications Zustand store.
 *
 * `ToastContainer` renders all current notifications as overlapping cards in
 * the bottom-right corner of the viewport. Each toast automatically dismisses
 * itself after `duration` milliseconds and can also be closed manually by the
 * user. Notifications enter via a slide-in animation and exit when removed
 * from the store.
 *
 * @module components/ui/Toast
 */
import { useEffect } from "react";
import { cn } from "@/lib/utils";
import {
  useNotificationsStore,
  type Notification,
  type NotificationLevel,
} from "@/store/notifications";

/** How long each toast stays visible before auto-dismissal. */
const TOAST_DURATION_MS = 4000;

/** Tailwind class sets keyed by notification severity level. */
const LEVEL_CLASSES: Record<NotificationLevel, string> = {
  info: "border-blue-400 bg-blue-50 text-blue-900 dark:bg-blue-950 dark:text-blue-100",
  success:
    "border-green-400 bg-green-50 text-green-900 dark:bg-green-950 dark:text-green-100",
  warning:
    "border-yellow-400 bg-yellow-50 text-yellow-900 dark:bg-yellow-950 dark:text-yellow-100",
  error: "border-red-400 bg-red-50 text-red-900 dark:bg-red-950 dark:text-red-100",
};

/** SVG path data for the close (×) icon. */
const CLOSE_PATH =
  "M6 18L18 6M6 6l12 12";

/**
 * A single dismissible toast card.
 *
 * Auto-removes itself from the store after `TOAST_DURATION_MS` and also
 * provides a close button for immediate dismissal.
 *
 * @param props.notification - The notification entry from the store.
 */
function Toast({ notification }: { notification: Notification }) {
  const remove = useNotificationsStore((s) => s.removeNotification);

  useEffect(() => {
    const timer = setTimeout(() => remove(notification.id), TOAST_DURATION_MS);
    return () => clearTimeout(timer);
  }, [notification.id, remove]);

  return (
    <div
      role="alert"
      aria-live="assertive"
      data-testid={`toast-${notification.id}`}
      className={cn(
        "flex items-start gap-3 rounded-lg border-l-4 px-4 py-3 shadow-lg text-sm",
        LEVEL_CLASSES[notification.level],
      )}
    >
      <span className="flex-1">{notification.message}</span>
      <button
        onClick={() => remove(notification.id)}
        aria-label="Dismiss notification"
        className="shrink-0 opacity-60 hover:opacity-100 transition-opacity"
        data-testid={`toast-close-${notification.id}`}
      >
        <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d={CLOSE_PATH} />
        </svg>
      </button>
    </div>
  );
}

/**
 * Fixed-position container that renders all active toast notifications.
 *
 * Mount this once near the root of the application (e.g. inside `App` or
 * `MainLayout`). It reads directly from the notifications store and requires
 * no props.
 */
export function ToastContainer() {
  const notifications = useNotificationsStore((s) => s.notifications);

  if (notifications.length === 0) return null;

  return (
    <div
      className="fixed bottom-4 right-4 z-[100] flex w-full max-w-sm flex-col gap-2"
      aria-label="Notifications"
      data-testid="toast-container"
    >
      {notifications.map((n) => (
        <Toast key={n.id} notification={n} />
      ))}
    </div>
  );
}
