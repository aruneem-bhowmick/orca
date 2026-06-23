import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import apiClient from "@/api/client";
import { ROUTES } from "@/lib/constants";
import { Button } from "@/components/ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import type { HealthStatus } from "@/api/types";

const serviceCards = [
  {
    title: "OrcaMind",
    description: "Meta-learning engine with task embeddings and model recommendation.",
    key: "orcamind" as const,
  },
  {
    title: "OrcaLab",
    description: "Experiment orchestration with hyperparameter sweeps and pruning.",
    key: "orcalab" as const,
  },
  {
    title: "OrcaNet",
    description: "Cross-domain knowledge transfer with retrieval and reasoning.",
    key: "orcanet" as const,
  },
];

export function Landing() {
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: async () => {
      const response = await apiClient.get<HealthStatus>("/health");
      return response.data;
    },
    refetchInterval: 30_000,
  });

  return (
    <div className="min-h-screen">
      <header className="border-b bg-card">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <span className="text-xl font-bold">Orca</span>
          <div className="flex gap-3">
            <Link to={ROUTES.LOGIN}>
              <Button variant="ghost">Sign in</Button>
            </Link>
            <Link to={ROUTES.REGISTER}>
              <Button>Get started</Button>
            </Link>
          </div>
        </div>
      </header>

      <section className="mx-auto max-w-7xl px-6 py-20 text-center">
        <h1 className="text-4xl font-bold tracking-tight sm:text-6xl">
          Meta-Learning Platform
        </h1>
        <p className="mt-6 text-lg leading-8 text-muted-foreground">
          Orca automates model selection, hyperparameter optimization, and cross-domain knowledge
          transfer — so you can go from dataset to deployed model faster.
        </p>
        <div className="mt-10 flex items-center justify-center gap-4">
          <Link to={ROUTES.REGISTER}>
            <Button size="lg">Start building</Button>
          </Link>
          <Link to={ROUTES.LOGIN}>
            <Button variant="outline" size="lg">
              Sign in
            </Button>
          </Link>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-6 py-12">
        <div className="grid gap-6 md:grid-cols-3" data-testid="service-cards">
          {serviceCards.map((svc) => (
            <Card key={svc.key}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg">{svc.title}</CardTitle>
                  {health && (
                    <span
                      className={`inline-block h-3 w-3 rounded-full ${
                        health.services[svc.key] ? "bg-green-500" : "bg-red-500"
                      }`}
                      data-testid={`status-${svc.key}`}
                    />
                  )}
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">{svc.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}
