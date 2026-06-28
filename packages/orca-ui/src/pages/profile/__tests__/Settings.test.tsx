import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor, fireEvent } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { Settings } from "@/pages/profile/Settings";
import apiClient from "@/api/client";
import { mockUser } from "@/test/mocks/handlers";

vi.mock("@/api/client", () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    patch: vi.fn(),
    interceptors: {
      request: { use: vi.fn(), handlers: [] },
      response: { use: vi.fn(), handlers: [] },
    },
  },
}));

// ---------------------------------------------------------------------------
// Auth store mock
// ---------------------------------------------------------------------------

const mockSetUser = vi.fn();

vi.mock("@/store/auth", () => ({
  useAuthStore: vi.fn(() => ({
    user: mockUser,
    setUser: mockSetUser,
  })),
}));

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(apiClient.patch).mockImplementation((url: string) => {
    if (url === "/auth/me") return Promise.resolve({ data: { ...mockUser, username: "newname" } });
    return Promise.reject(new Error(`Unexpected PATCH: ${url}`));
  });
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("Settings", () => {
  it("renders the page heading", () => {
    render(<Settings />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Settings");
  });

  it("displays the current user email in read-only form", () => {
    render(<Settings />);
    expect(screen.getByTestId("email-display")).toHaveTextContent(mockUser.email);
  });

  it("pre-fills the username input with the current username", () => {
    render(<Settings />);
    expect(screen.getByTestId("username-input")).toHaveValue(mockUser.username);
  });

  it("renders the Save Changes button", () => {
    render(<Settings />);
    expect(screen.getByTestId("save-btn")).toBeInTheDocument();
  });

  it("renders notification preference toggles", () => {
    render(<Settings />);
    expect(screen.getByTestId("pref-notify_experiment_complete")).toBeInTheDocument();
    expect(screen.getByTestId("pref-notify_sweep_complete")).toBeInTheDocument();
    expect(screen.getByTestId("pref-notify_transfer_scored")).toBeInTheDocument();
  });

  it("renders the OAuth connections section", () => {
    render(<Settings />);
    expect(screen.getByTestId("oauth-connections")).toBeInTheDocument();
  });

  it("shows 'no OAuth providers' when user has no oauth_provider", () => {
    render(<Settings />);
    expect(screen.getByTestId("oauth-connections")).toHaveTextContent(
      "No OAuth providers connected",
    );
  });

  it("calls PATCH /auth/me with updated username on save", async () => {
    render(<Settings />);
    fireEvent.change(screen.getByTestId("username-input"), {
      target: { value: "newname" },
    });
    fireEvent.click(screen.getByTestId("save-btn"));

    await waitFor(() => {
      expect(apiClient.patch).toHaveBeenCalledWith(
        "/auth/me",
        expect.objectContaining({ username: "newname" }),
      );
    });
  });

  it("calls setUser with the updated user on successful save", async () => {
    render(<Settings />);
    fireEvent.click(screen.getByTestId("save-btn"));

    await waitFor(() => {
      expect(mockSetUser).toHaveBeenCalledWith(
        expect.objectContaining({ username: "newname" }),
      );
    });
  });

  it("shows a success message after saving", async () => {
    render(<Settings />);
    fireEvent.click(screen.getByTestId("save-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("settings-success")).toBeInTheDocument();
    });
  });

  it("shows an error message when PATCH /auth/me fails", async () => {
    vi.mocked(apiClient.patch).mockRejectedValue(new Error("network"));
    render(<Settings />);
    fireEvent.click(screen.getByTestId("save-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("settings-error")).toBeInTheDocument();
    });
  });

  it("toggles a notification preference when its switch is clicked", () => {
    render(<Settings />);
    const toggle = screen.getByTestId("pref-notify_experiment_complete");
    // Default is true (on), click should toggle to false
    const initialState = toggle.getAttribute("aria-checked");
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-checked")).not.toBe(initialState);
  });

  it("includes updated preferences in the PATCH payload", async () => {
    render(<Settings />);
    // Toggle one preference off
    fireEvent.click(screen.getByTestId("pref-notify_experiment_complete"));
    fireEvent.click(screen.getByTestId("save-btn"));

    await waitFor(() => {
      expect(apiClient.patch).toHaveBeenCalledWith(
        "/auth/me",
        expect.objectContaining({
          preferences: expect.objectContaining({
            notify_experiment_complete: false,
          }),
        }),
      );
    });
  });

  it("merges notification preferences with other existing preferences in the PATCH payload", async () => {
    const userWithExtraPrefs = {
      ...mockUser,
      preferences: {
        theme: "dark",
        notify_experiment_complete: true,
      },
    };
    const { useAuthStore } = await import("@/store/auth");
    vi.mocked(useAuthStore).mockReturnValue({
      user: userWithExtraPrefs,
      setUser: mockSetUser,
    } as any);

    render(<Settings />);

    fireEvent.click(screen.getByTestId("pref-notify_experiment_complete"));
    fireEvent.click(screen.getByTestId("save-btn"));

    await waitFor(() => {
      expect(apiClient.patch).toHaveBeenCalledWith(
        "/auth/me",
        expect.objectContaining({
          preferences: expect.objectContaining({
            theme: "dark",
            notify_experiment_complete: false,
          }),
        }),
      );
    });
  });

  it("resets local state (clears username and restores prefs to default) when user becomes absent", async () => {
    const { useAuthStore } = await import("@/store/auth");
    
    let currentUser: any = mockUser;
    vi.mocked(useAuthStore).mockImplementation(() => ({
      user: currentUser,
      setUser: mockSetUser,
    }));

    const { rerender } = render(<Settings />);
    expect(screen.getByTestId("username-input")).toHaveValue(mockUser.username);

    currentUser = null;
    rerender(<Settings />);

    expect(screen.getByTestId("username-input")).toHaveValue("");
    const toggle = screen.getByTestId("pref-notify_experiment_complete");
    expect(toggle.getAttribute("aria-checked")).toBe("false");
  });
});
