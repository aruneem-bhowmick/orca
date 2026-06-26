import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useAuthStore } from "@/store/auth";
import { exchangeOAuthCode, getMe } from "@/api/auth";
import { ROUTES } from "@/lib/constants";

/**
 * OAuth callback handler page.
 *
 * Rendered at `/oauth/callback` after an OAuth provider (Google or
 * GitHub) redirects back with authorization parameters. Reads the
 * `provider` and optional `error` from the URL query string, then
 * exchanges the authorization code for an access token via the BFF.
 * On success, stores the token and user profile in the auth store
 * and redirects to the dashboard. On failure, displays an error
 * message with a link back to the login page.
 */
export function OAuthCallback() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function handleCallback() {
      const provider = searchParams.get("provider");
      const errParam = searchParams.get("error");

      if (errParam) {
        setError(errParam);
        return;
      }

      if (!provider) {
        setError("Missing OAuth provider.");
        return;
      }

      const code = searchParams.get("code");
      if (!code) {
        setError("Missing authorization code.");
        return;
      }

      try {
        const tokenResponse = await exchangeOAuthCode(provider, searchParams);
        useAuthStore.getState().setToken(tokenResponse.access_token);
        const user = await getMe();
        useAuthStore.getState().setAuth(user, tokenResponse.access_token);
        navigate(ROUTES.DASHBOARD, { replace: true });
      } catch {
        setError("Failed to complete authentication.");
        useAuthStore.getState().clearAuth();
      }
    }

    handleCallback();
  }, [searchParams, navigate]);

  if (error) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-center">
          <p className="text-destructive" data-testid="oauth-error">
            {error}
          </p>
          <a href={ROUTES.LOGIN} className="mt-4 inline-block text-primary hover:underline">
            Back to login
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen items-center justify-center">
      <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
    </div>
  );
}
