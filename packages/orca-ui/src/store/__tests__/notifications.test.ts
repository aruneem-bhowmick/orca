import { describe, it, expect, beforeEach } from "vitest";
import { useNotificationsStore } from "@/store/notifications";

describe("useNotificationsStore", () => {
  beforeEach(() => {
    useNotificationsStore.getState().clearNotifications();
  });

  it("starts with an empty notification list", () => {
    expect(useNotificationsStore.getState().notifications).toHaveLength(0);
  });

  it("addNotification appends a notification with a unique id", () => {
    useNotificationsStore.getState().addNotification("Hello world");
    const { notifications } = useNotificationsStore.getState();
    expect(notifications).toHaveLength(1);
    expect(notifications[0].message).toBe("Hello world");
    expect(notifications[0].level).toBe("info");
    expect(notifications[0].id).toBeTruthy();
  });

  it("addNotification uses the supplied level", () => {
    useNotificationsStore.getState().addNotification("Error!", "error");
    const { notifications } = useNotificationsStore.getState();
    expect(notifications[0].level).toBe("error");
  });

  it("addNotification generates unique IDs for concurrent calls", () => {
    const { addNotification } = useNotificationsStore.getState();
    addNotification("First");
    addNotification("Second");
    const { notifications } = useNotificationsStore.getState();
    expect(notifications).toHaveLength(2);
    expect(notifications[0].id).not.toBe(notifications[1].id);
  });

  it("removeNotification removes only the matching id", () => {
    const { addNotification, removeNotification } =
      useNotificationsStore.getState();
    addNotification("Keep me");
    addNotification("Remove me");
    const { notifications } = useNotificationsStore.getState();
    const removeId = notifications[1].id;

    removeNotification(removeId);
    const after = useNotificationsStore.getState().notifications;
    expect(after).toHaveLength(1);
    expect(after[0].message).toBe("Keep me");
  });

  it("removeNotification is a no-op for an unknown id", () => {
    useNotificationsStore.getState().addNotification("Stays");
    useNotificationsStore.getState().removeNotification("nonexistent-id");
    expect(useNotificationsStore.getState().notifications).toHaveLength(1);
  });

  it("clearNotifications removes all entries", () => {
    const { addNotification, clearNotifications } =
      useNotificationsStore.getState();
    addNotification("A");
    addNotification("B");
    addNotification("C");
    clearNotifications();
    expect(useNotificationsStore.getState().notifications).toHaveLength(0);
  });

  it("supports all four notification levels", () => {
    const store = useNotificationsStore.getState();
    store.addNotification("info msg", "info");
    store.addNotification("success msg", "success");
    store.addNotification("warning msg", "warning");
    store.addNotification("error msg", "error");
    const levels = useNotificationsStore
      .getState()
      .notifications.map((n) => n.level);
    expect(levels).toEqual(["info", "success", "warning", "error"]);
  });
});
