import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useAuthStore } from "@/store/auth";
import { exchangeOAuthCode, getMe } from "@/api/auth";
import { ROUTES } from "@/lib/constants";

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
