import { useQuery } from "@tanstack/react-query";
import apiClient from "@/api/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import type { HealthStatus } from "@/api/types";

export function Dashboard() {
  const { data: health, isLoading } = useQuery({
    queryKey: ["dashboard-health"],
    queryFn: async () => {
      const response = await apiClient.get<HealthStatus>("/health", {
        validateStatus: (status) => (status >= 200 && status < 300) || status === 503,
      });
      return response.data;
    },
    refetchInterval: 15_000,
  });

  return (
    <div>
      <h1 className="text-2xl font-bold">Dashboard</h1>
      <p className="mt-1 text-muted-foreground">Overview of the Orca platform.</p>

      <div className="mt-6 grid gap-6 md:grid-cols-3" data-testid="dashboard-cards">
        {isLoading ? (
          <p className="text-muted-foreground">Loading...</p>
        ) : (
          <>
            <Card>
              <CardHeader>
                <CardTitle className="text-base">OrcaMind</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Status:{" "}
                  <span
                    className={
                      health?.services.orcamind ? "text-green-600" : "text-red-600"
                    }
                  >
                    {health?.services.orcamind ? "Online" : "Offline"}
                  </span>
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-base">OrcaLab</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Status:{" "}
                  <span
                    className={
                      health?.services.orcalab ? "text-green-600" : "text-red-600"
                    }
                  >
                    {health?.services.orcalab ? "Online" : "Offline"}
                  </span>
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-base">OrcaNet</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Status:{" "}
                  <span
                    className={
                      health?.services.orcanet ? "text-green-600" : "text-red-600"
                    }
                  >
                    {health?.services.orcanet ? "Online" : "Offline"}
                  </span>
                </p>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  );
}
