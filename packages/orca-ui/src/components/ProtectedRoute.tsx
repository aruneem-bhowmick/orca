import { Navigate } from "react-router-dom";
import { useAuth } from "@/hooks/useAuth";
import { ROUTES } from "@/lib/constants";

/**
 * Route guard component that restricts access to authenticated users.
 *
 * While the initial auth check is in progress (`isLoading`), renders a
 * centered loading spinner. Once loading is complete, either renders
 * the child components (authenticated) or redirects to the login page
 * (unauthenticated).
 *
 * @param props.children - The protected page content to render.
 */
export function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center" data-testid="auth-loading">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to={ROUTES.LOGIN} replace />;
  }

  return <>{children}</>;
}
