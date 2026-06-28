import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ROUTES } from "@/lib/constants";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { MainLayout } from "@/components/layout/MainLayout";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { NotFound } from "@/components/NotFound";
import { ToastContainer } from "@/components/ui/Toast";
import { RouteProgress } from "@/components/ui/RouteProgress";
import { Landing } from "@/pages/Landing";
import { Login } from "@/pages/Login";
import { Register } from "@/pages/Register";
import { OAuthCallback } from "@/pages/OAuthCallback";
import { Dashboard } from "@/pages/Dashboard";
import { TaskList } from "@/pages/orcamind/TaskList";
import { TaskDetail } from "@/pages/orcamind/TaskDetail";
import { Recommendations } from "@/pages/orcamind/Recommendations";
import { ExperimentList } from "@/pages/orcalab/ExperimentList";
import { ExperimentDetail } from "@/pages/orcalab/ExperimentDetail";
import { SweepManager } from "@/pages/orcalab/SweepManager";
import { TransferExplorer } from "@/pages/orcanet/TransferExplorer";
import { RetrievalView } from "@/pages/orcanet/RetrievalView";
import { ActivityLog } from "@/pages/history/ActivityLog";
import { MyTasks } from "@/pages/history/MyTasks";
import { MyExperiments } from "@/pages/history/MyExperiments";
import { Settings } from "@/pages/profile/Settings";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

/**
 * Placeholder component for routes that have not yet been implemented.
 * Renders the page title and a "Coming soon" message.
 *
 * @param props.title - The page heading to display.
 */
function PlaceholderPage({ title }: { title: string }) {
  return (
    <div>
      <h1 className="text-2xl font-bold">{title}</h1>
      <p className="mt-1 text-muted-foreground">Coming soon.</p>
    </div>
  );
}

/**
 * Root application component.
 *
 * Wraps the entire app in `QueryClientProvider`, `BrowserRouter`, and an
 * `ErrorBoundary` that catches uncaught rendering errors and displays a
 * recovery screen. Inside the router, `RouteProgress` renders a top-of-page
 * progress bar during navigation transitions and `ToastContainer` displays
 * toast notifications driven by the Zustand notifications store.
 *
 * Public routes (landing, login, register, OAuth callback) are accessible
 * without authentication. Protected routes are wrapped in `ProtectedRoute`
 * and rendered inside `MainLayout` which provides the sidebar and header.
 * An explicit catch-all route renders the `NotFound` component for any URL
 * that does not match the route tree.
 */
export function App() {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <RouteProgress />
          <Routes>
            {/* Public routes */}
            <Route path={ROUTES.HOME} element={<Landing />} />
            <Route path={ROUTES.LOGIN} element={<Login />} />
            <Route path={ROUTES.REGISTER} element={<Register />} />
            <Route path={ROUTES.OAUTH_CALLBACK} element={<OAuthCallback />} />

            {/* Protected routes under MainLayout */}
            <Route
              element={
                <ProtectedRoute>
                  <MainLayout />
                </ProtectedRoute>
              }
            >
              {/* Dashboard overview */}
              <Route path={ROUTES.DASHBOARD} element={<Dashboard />} />

              {/* OrcaMind routes */}
              <Route path={ROUTES.ORCAMIND_TASKS} element={<TaskList />} />
              <Route
                path={ROUTES.ORCAMIND_TASK_DETAIL}
                element={<TaskDetail />}
              />
              <Route
                path={ROUTES.ORCAMIND_RECOMMENDATIONS}
                element={<Recommendations />}
              />

              {/* OrcaLab routes */}
              <Route
                path={ROUTES.ORCALAB_EXPERIMENTS}
                element={<ExperimentList />}
              />
              <Route
                path={ROUTES.ORCALAB_EXPERIMENT_DETAIL}
                element={<ExperimentDetail />}
              />
              <Route
                path={ROUTES.ORCALAB_SWEEPS}
                element={<SweepManager />}
              />

              {/* OrcaNet routes */}
              <Route
                path={ROUTES.ORCANET_TRANSFER}
                element={<TransferExplorer />}
              />
              <Route
                path={ROUTES.ORCANET_RETRIEVE}
                element={<RetrievalView />}
              />

              {/* History routes */}
              <Route
                path={ROUTES.HISTORY}
                element={<ActivityLog />}
              />
              <Route
                path={ROUTES.HISTORY_TASKS}
                element={<MyTasks />}
              />
              <Route
                path={ROUTES.HISTORY_EXPERIMENTS}
                element={<MyExperiments />}
              />

              {/* Bookmarks */}
              <Route
                path={ROUTES.BOOKMARKS}
                element={<PlaceholderPage title="Bookmarks" />}
              />

              {/* Profile */}
              <Route
                path={ROUTES.PROFILE}
                element={<Settings />}
              />
            </Route>

            {/* 404 catch-all */}
            <Route path="*" element={<NotFound />} />
          </Routes>

          <ToastContainer />
        </BrowserRouter>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}
