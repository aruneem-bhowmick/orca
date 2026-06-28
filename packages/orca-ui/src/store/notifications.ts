/**
 * Zustand store for application-wide toast notifications.
 *
 * Notifications are displayed non-intrusively for API errors and
 * informational messages that do not block the user workflow. Each
 * notification carries an auto-generated ID, a severity level, and a
 * message string. Callers add notifications via `addNotification` and the
 * `ToastContainer` component removes them after a configurable timeout.
 *
 * @module store/notifications
 */
import { create } from "zustand";

/** Severity level that determines the visual appearance of a notification. */
export type NotificationLevel = "info" | "success" | "warning" | "error";

/** A single toast notification entry. */
export interface Notification {
  /** Unique identifier, assigned automatically by `addNotification`. */
  id: string;
  /** Message text to display to the user. */
  message: string;
  /** Severity level controlling colour and icon. */
  level: NotificationLevel;
}

/** Shape of the notification store and its action methods. */
export interface NotificationsState {
  /** All currently visible notifications, in insertion order. */
  notifications: Notification[];
  /**
   * Add a new notification.
   *
   * @param message - The message to display.
   * @param level - The severity level (defaults to `"info"`).
   */
  addNotification: (message: string, level?: NotificationLevel) => void;
  /**
   * Remove a notification by its ID.
   *
   * @param id - The notification to remove.
   */
  removeNotification: (id: string) => void;
  /** Remove all notifications at once. */
  clearNotifications: () => void;
}

/**
 * Zustand store for toast notifications.
 *
 * Use `addNotification` from anywhere in the app (including outside React
 * via `useNotificationsStore.getState().addNotification(…)`).
 */
export const useNotificationsStore = create<NotificationsState>((set) => ({
  notifications: [],

  addNotification: (message, level = "info") => {
    const id = typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    set((state) => ({
      notifications: [...state.notifications, { id, message, level }],
    }));
  },

  removeNotification: (id) =>
    set((state) => ({
      notifications: state.notifications.filter((n) => n.id !== id),
    })),

  clearNotifications: () => set({ notifications: [] }),
}));
