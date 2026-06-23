import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ROUTES } from "@/lib/constants";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { MainLayout } from "@/components/layout/MainLayout";
import { Landing } from "@/pages/Landing";
import { Login } from "@/pages/Login";
import { Register } from "@/pages/Register";
import { OAuthCallback } from "@/pages/OAuthCallback";
import { Dashboard } from "@/pages/Dashboard";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function PlaceholderPage({ title }: { title: string }) {
  return (
    <div>
      <h1 className="text-2xl font-bold">{title}</h1>
      <p className="mt-1 text-muted-foreground">Coming soon.</p>
    </div>
  );
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          {/* Public routes */}
          <Route path={ROUTES.HOME} element={<Landing />} />
          <Route path={ROUTES.LOGIN} element={<Login />} />
          <Route path={ROUTES.REGISTER} element={<Register />} />
          <Route path={ROUTES.OAUTH_CALLBACK} element={<OAuthCallback />} />

          {/* Protected routes */}
          <Route
            element={
              <ProtectedRoute>
                <MainLayout />
              </ProtectedRoute>
            }
          >
            <Route path={ROUTES.DASHBOARD} element={<Dashboard />} />
            <Route path={ROUTES.TASKS} element={<PlaceholderPage title="Tasks" />} />
            <Route path={ROUTES.EXPERIMENTS} element={<PlaceholderPage title="Experiments" />} />
            <Route path={ROUTES.SWEEPS} element={<PlaceholderPage title="Sweeps" />} />
            <Route path={ROUTES.TRANSFERS} element={<PlaceholderPage title="Transfers" />} />
            <Route path={ROUTES.HISTORY} element={<PlaceholderPage title="History" />} />
            <Route path={ROUTES.BOOKMARKS} element={<PlaceholderPage title="Bookmarks" />} />
            <Route path={ROUTES.SETTINGS} element={<PlaceholderPage title="Settings" />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
