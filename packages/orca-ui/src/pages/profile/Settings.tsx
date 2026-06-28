import { useEffect, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import apiClient from "@/api/client";
import { useAuthStore } from "@/store/auth";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import type { ProfileUpdate, User } from "@/api/types";

// ---------------------------------------------------------------------------
// Preference toggle
// ---------------------------------------------------------------------------

/**
 * A labelled toggle switch for a single boolean preference key.
 *
 * @param props.label - Human-readable label for the preference.
 * @param props.checked - Current toggle state.
 * @param props.onChange - Callback invoked when the toggle changes.
 * @param props.testId - `data-testid` applied to the checkbox input.
 */
function PreferenceToggle({
  label,
  checked,
  onChange,
  testId,
}: {
  label: string;
  checked: boolean;
  onChange: (value: boolean) => void;
  testId: string;
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-4">
      <span className="text-sm">{label}</span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ${checked ? "bg-primary" : "bg-input"}`}
        data-testid={testId}
      >
        <span
          className={`pointer-events-none inline-block h-5 w-5 rounded-full bg-background shadow-lg ring-0 transition-transform ${checked ? "translate-x-5" : "translate-x-0"}`}
        />
      </button>
    </label>
  );
}

// ---------------------------------------------------------------------------
// Settings page
// ---------------------------------------------------------------------------

/** Keys in `user.preferences` that represent notification toggles. */
const NOTIFICATION_PREFS: { key: string; label: string }[] = [
  { key: "notify_experiment_complete", label: "Experiment completed" },
  { key: "notify_sweep_complete", label: "Sweep completed" },
  { key: "notify_transfer_scored", label: "Transfer scored" },
];

/**
 * Returns the boolean value of a preference key from the user's preferences
 * object, defaulting to `true` when the key is absent.
 *
 * @param prefs - The user's preferences map (may be null).
 * @param key - The preference key to read.
 * @returns Boolean preference value.
 */
function getPref(prefs: Record<string, unknown> | null, key: string): boolean {
  if (!prefs || !(key in prefs)) return true;
  return Boolean(prefs[key]);
}

/**
 * User Settings page.
 *
 * Provides two sections:
 * 1. **Profile** — editable username field sent to `PATCH /auth/me` on save.
 * 2. **Notification preferences** — boolean toggles stored in
 *    `user.preferences` and persisted via the same endpoint.
 *
 * The current user is read from the Zustand auth store; the store's
 * `setUser` action is called on successful save to keep the UI in sync
 * without a full page reload.
 *
 * A read-only **Account** section shows connected OAuth providers derived
 * from the user's `oauth_provider` field (when present in the BFF response).
 */
export function Settings() {
  const { user, setUser } = useAuthStore();

  const [username, setUsername] = useState(user?.username ?? "");
  const [prefs, setPrefs] = useState<Record<string, boolean>>(() => {
    const p = user?.preferences ?? null;
    return Object.fromEntries(
      NOTIFICATION_PREFS.map(({ key }) => [key, getPref(p as Record<string, unknown> | null, key)]),
    );
  });

  const [saveSuccess, setSaveSuccess] = useState(false);

  // Sync local state when the store user changes (e.g. after mount restore).
  useEffect(() => {
    if (user) {
      setUsername(user.username);
      setPrefs(
        Object.fromEntries(
          NOTIFICATION_PREFS.map(({ key }) => [
            key,
            getPref(user.preferences as Record<string, unknown> | null, key),
          ]),
        ),
      );
    }
  }, [user]);

  const { mutate: save, isPending: isSaving, isError: saveError } = useMutation({
    mutationFn: async () => {
      const payload: ProfileUpdate = {
        username: username.trim(),
        preferences: prefs as Record<string, unknown>,
      };
      const res = await apiClient.patch<User>("/auth/me", payload);
      return res.data;
    },
    onSuccess: (updated) => {
      setUser(updated);
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    },
  });

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSaveSuccess(false);
    save();
  }

  /** Update a single boolean preference key. */
  function handlePrefChange(key: string, value: boolean) {
    setPrefs((prev) => ({ ...prev, [key]: value }));
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Settings</h1>
        <p className="mt-1 text-muted-foreground">Manage your profile and notification preferences.</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Profile section */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Profile</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="mb-1 block text-sm font-medium text-muted-foreground">
                Email
              </label>
              <p className="text-sm" data-testid="email-display">
                {user?.email ?? "—"}
              </p>
            </div>
            <Input
              label="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              data-testid="username-input"
            />
          </CardContent>
        </Card>

        {/* Notification preferences */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Notification Preferences</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {NOTIFICATION_PREFS.map(({ key, label }) => (
              <PreferenceToggle
                key={key}
                label={label}
                checked={prefs[key] ?? true}
                onChange={(v) => handlePrefChange(key, v)}
                testId={`pref-${key}`}
              />
            ))}
          </CardContent>
        </Card>

        {/* Save */}
        {saveError && (
          <p className="text-sm text-destructive" data-testid="settings-error">
            Failed to save settings. Please try again.
          </p>
        )}
        {saveSuccess && (
          <p className="text-sm text-green-600" data-testid="settings-success">
            Settings saved.
          </p>
        )}
        <Button type="submit" disabled={isSaving} data-testid="save-btn">
          {isSaving ? "Saving…" : "Save Changes"}
        </Button>
      </form>

      {/* OAuth connections */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Connected Accounts</CardTitle>
        </CardHeader>
        <CardContent data-testid="oauth-connections">
          {(user as (User & { oauth_provider?: string | null }) | null)?.oauth_provider ? (
            <p className="text-sm">
              Connected via{" "}
              <span className="font-medium capitalize">
                {(user as User & { oauth_provider?: string })!.oauth_provider}
              </span>
            </p>
          ) : (
            <p className="text-sm text-muted-foreground">
              No OAuth providers connected. Sign in with Google or GitHub on the login page
              to link an account.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
