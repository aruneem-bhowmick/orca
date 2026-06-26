import { Link, Navigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import apiClient from "@/api/client";
import { ROUTES } from "@/lib/constants";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { useAuth } from "@/hooks/useAuth";
import type { HealthStatus, DashboardStats } from "@/api/types";

/**
 * Service card definitions for the landing page grid.
 * Each card displays the service name, a brief description,
 * an icon path, and a health status indicator.
 */
const serviceCards = [
  {
    title: "OrcaMind",
    description:
      "Meta-learning engine with task embeddings and model recommendation.",
    key: "orcamind" as const,
    icon: "M9 3v2H6a1 1 0 0 0-1 1v3H3v2h2v4H3v2h2v3a1 1 0 0 0 1 1h3v2h2v-2h4v2h2v-2h3a1 1 0 0 0 1-1v-3h2v-2h-2v-4h2V9h-2V6a1 1 0 0 0-1-1h-3V3h-2v2h-4V3H9z",
  },
  {
    title: "OrcaLab",
    description:
      "Experiment orchestration with hyperparameter sweeps and pruning.",
    key: "orcalab" as const,
    icon: "M10 2v6.292a4 4 0 0 0-1.17 1.17L4 16.586V20a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3.414l-4.83-7.124A4 4 0 0 0 14 8.292V2h-4z",
  },
  {
    title: "OrcaNet",
    description:
      "Cross-domain knowledge transfer with retrieval and reasoning.",
    key: "orcanet" as const,
    icon: "M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05-4.11A2.99 2.99 0 0 0 18 8a3 3 0 1 0-3-3c0 .24.04.47.09.7L8.04 9.81A2.99 2.99 0 0 0 6 9a3 3 0 0 0 0 6c.79 0 1.5-.31 2.04-.81l7.12 4.16c-.05.21-.08.43-.08.65a2.92 2.92 0 0 0 2.92 2.92A2.92 2.92 0 0 0 21 19a2.92 2.92 0 0 0-3-2.92z",
  },
];

/**
 * Public landing page displayed to unauthenticated users at `/`.
 *
 * Authenticated users are automatically redirected to `/dashboard`.
 *
 * The page consists of four sections:
 * 1. **Hero** — headline, description, and CTAs for registration and sign-in.
 * 2. **Service cards** — three cards for OrcaMind, OrcaLab, OrcaNet
 *    with icons, descriptions, and health status indicators.
 * 3. **Live stats** — three counters showing tasks registered, experiments
 *    run, and transfers scored, fetched from `GET /dashboard/stats` with
 *    a 60-second refetch interval.
 * 4. **Footer** — links to documentation and the GitHub repository.
 *
 * The layout is responsive: cards and stats render in a 3-column grid
 * on desktop and stack vertically on mobile.
 */
export function Landing() {
  const { isAuthenticated, isLoading } = useAuth();

  const shouldFetchPublicData = !isLoading && !isAuthenticated;

  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: async () => {
      const response = await apiClient.get<HealthStatus>("/health");
      return response.data;
    },
    refetchInterval: 30_000,
    enabled: shouldFetchPublicData,
  });

  const { data: stats } = useQuery({
    queryKey: ["dashboard-stats"],
    queryFn: async () => {
      const response = await apiClient.get<DashboardStats>("/dashboard/stats");
      return response.data;
    },
    refetchInterval: 60_000,
    enabled: shouldFetchPublicData,
  });

  if (isLoading) {
    return null;
  }

  if (isAuthenticated) {
    return <Navigate to={ROUTES.DASHBOARD} replace />;
  }

  return (
    <div className="min-h-screen">
      {/* Navigation header */}
      <header className="border-b bg-card">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <span className="text-xl font-bold">Orca</span>
          <div className="flex gap-3">
            <Link to={ROUTES.LOGIN}>
              <Button variant="ghost">Sign In</Button>
            </Link>
            <Link to={ROUTES.REGISTER}>
              <Button>Get Started</Button>
            </Link>
          </div>
        </div>
      </header>

      {/* Hero section */}
      <section className="mx-auto max-w-7xl px-6 py-20 text-center">
        <h1 className="text-4xl font-bold tracking-tight sm:text-6xl">
          Orca: Meta-Learning Platform
        </h1>
        <p className="mt-6 text-lg leading-8 text-muted-foreground">
          Orca automates model selection, hyperparameter optimization, and
          cross-domain knowledge transfer — so you can go from dataset to
          deployed model faster.
        </p>
        <div className="mt-10 flex items-center justify-center gap-4">
          <Link to={ROUTES.REGISTER}>
            <Button size="lg">Get Started</Button>
          </Link>
          <Link to={ROUTES.LOGIN}>
            <Button variant="outline" size="lg">
              Sign In
            </Button>
          </Link>
        </div>
      </section>

      {/* Service cards */}
      <section className="mx-auto max-w-7xl px-6 py-12">
        <div
          className="grid gap-6 md:grid-cols-3"
          data-testid="service-cards"
        >
          {serviceCards.map((svc) => (
            <Card key={svc.key}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <svg
                      className="h-6 w-6 text-primary"
                      viewBox="0 0 24 24"
                      fill="currentColor"
                      data-testid={`icon-${svc.key}`}
                    >
                      <path d={svc.icon} />
                    </svg>
                    <CardTitle className="text-lg">{svc.title}</CardTitle>
                  </div>
                  {health && (
                    <span
                      className={`inline-block h-3 w-3 rounded-full ${
                        health.services[svc.key]
                          ? "bg-green-500"
                          : "bg-red-500"
                      }`}
                      data-testid={`status-${svc.key}`}
                    />
                  )}
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  {svc.description}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Live stats */}
      <section className="mx-auto max-w-7xl px-6 py-12" data-testid="live-stats">
        <h2 className="mb-8 text-center text-2xl font-bold">
          Platform at a Glance
        </h2>
        <div className="grid gap-6 md:grid-cols-3">
          <Card>
            <CardContent className="pt-6 text-center">
              <p
                className="text-4xl font-bold text-primary"
                data-testid="stat-tasks"
              >
                {stats?.tasks_registered ?? "—"}
              </p>
              <p className="mt-2 text-sm text-muted-foreground">
                Tasks Registered
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6 text-center">
              <p
                className="text-4xl font-bold text-primary"
                data-testid="stat-experiments"
              >
                {stats?.experiments_run ?? "—"}
              </p>
              <p className="mt-2 text-sm text-muted-foreground">
                Experiments Run
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6 text-center">
              <p
                className="text-4xl font-bold text-primary"
                data-testid="stat-transfers"
              >
                {stats?.transfers_scored ?? "—"}
              </p>
              <p className="mt-2 text-sm text-muted-foreground">
                Transfers Scored
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Footer */}
      <footer
        className="border-t bg-card py-8"
        data-testid="landing-footer"
      >
        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-6 sm:flex-row">
          <p className="text-sm text-muted-foreground">
            Orca Meta-Learning Platform
          </p>
          <div className="flex gap-6">
            <a
              href="https://github.com/aruneem-bhowmick/orca/tree/main/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-muted-foreground hover:text-foreground"
              data-testid="docs-link"
            >
              Documentation
            </a>
            <a
              href="https://github.com/aruneem-bhowmick/orca"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-muted-foreground hover:text-foreground"
              data-testid="github-link"
            >
              GitHub
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
