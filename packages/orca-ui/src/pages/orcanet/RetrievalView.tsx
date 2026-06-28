import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import apiClient from "@/api/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { ROUTES } from "@/lib/constants";
import type { RetrieveRequest, RetrieveResult } from "@/api/types";

// ---------------------------------------------------------------------------
// Result card
// ---------------------------------------------------------------------------

/**
 * A single retrieval result card showing the matched task's name,
 * domain, type, similarity score, and a link to its OrcaMind detail page.
 *
 * @param props.result - The retrieval result to display.
 * @param props.onView - Callback invoked when the user navigates to the task.
 */
function RetrievalResultCard({
  result,
  onView,
}: {
  result: RetrieveResult;
  onView: (taskId: string) => void;
}) {
  const pct = Math.round(result.similarity_score * 100);
  return (
    <Card data-testid={`result-card-${result.task_id}`}>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">{result.name}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap gap-3 text-sm">
          <span>
            <span className="text-muted-foreground">Domain: </span>
            <span className="font-medium" data-testid="result-domain">
              {result.domain}
            </span>
          </span>
          <span>
            <span className="text-muted-foreground">Type: </span>
            <span className="font-medium">{result.task_type}</span>
          </span>
          <span>
            <span className="text-muted-foreground">Similarity: </span>
            <span
              className="font-medium text-primary"
              data-testid="result-score"
            >
              {pct}%
            </span>
          </span>
        </div>
        {/* Similarity bar */}
        <div className="h-2 w-full overflow-hidden rounded-full bg-muted" aria-hidden="true">
          <div
            className="h-full rounded-full bg-primary transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => onView(result.task_id)}
          data-testid="view-task-btn"
        >
          View Details
        </Button>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// RetrievalView page
// ---------------------------------------------------------------------------

/**
 * OrcaNet Retrieval View page.
 *
 * Accepts a free-text natural-language description of a target task and
 * searches for semantically similar tasks in the OrcaMind registry via
 * `POST /orcanet/retrieve`. Results are displayed as cards ordered by
 * similarity score, each linking to the corresponding OrcaMind task
 * detail page.
 */
export function RetrievalView() {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<RetrieveResult[]>([]);
  const [searched, setSearched] = useState(false);

  const { mutate: search, isPending: isSearching, isError } = useMutation({
    mutationFn: async () => {
      const payload: RetrieveRequest = { query: query.trim() };
      const res = await apiClient.post<RetrieveResult[]>("/orcanet/retrieve", payload);
      return res.data;
    },
    onSuccess: (data) => {
      setResults(data);
      setSearched(true);
    },
    onError: () => {
      setResults([]);
      setSearched(true);
    },
  });

  /** Navigate to the OrcaMind task detail page for the given task ID. */
  function handleView(taskId: string) {
    navigate(`${ROUTES.ORCAMIND_TASKS}/${taskId}`);
  }

  /**
   * Handles the search form submission, preventing the default page reload
   * and firing the retrieval mutation when the query is non-empty.
   *
   * @param e - The React form submission event.
   */
  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (query.trim()) search();
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Task Retrieval</h1>
        <p className="mt-1 text-muted-foreground">
          Find similar OrcaMind tasks using a natural-language description.
        </p>
      </div>

      {/* Search form */}
      <form onSubmit={handleSubmit} className="flex items-end gap-3">
        <div className="flex-1">
          <Input
            label="Task description"
            placeholder="e.g. image classification on medical X-rays with 10 classes…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                if (query.trim()) search();
              }
            }}
            data-testid="retrieval-query-input"
          />
        </div>
        <Button
          type="submit"
          disabled={!query.trim() || isSearching}
          data-testid="retrieval-search-btn"
        >
          {isSearching ? "Searching…" : "Search"}
        </Button>
      </form>

      {/* Error */}
      {isError && (
        <p className="text-sm text-destructive" data-testid="retrieval-error">
          Failed to retrieve tasks. Please try again.
        </p>
      )}

      {/* Results */}
      {searched && !isError && results.length === 0 && (
        <p className="text-muted-foreground" data-testid="retrieval-empty">
          No matching tasks found.
        </p>
      )}

      {results.length > 0 && (
        <div className="space-y-3" data-testid="retrieval-results">
          <p className="text-sm text-muted-foreground">
            {results.length} result{results.length !== 1 ? "s" : ""} found
          </p>
          {results.map((r) => (
            <RetrievalResultCard key={r.task_id} result={r} onView={handleView} />
          ))}
        </div>
      )}
    </div>
  );
}
