import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import apiClient from "@/api/client";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { formatDate } from "@/lib/utils";
import { ROUTES } from "@/lib/constants";
import type { Task, EmbedTaskRequest } from "@/api/types";

type SortKey = keyof Pick<
  Task,
  "name" | "domain" | "task_type" | "n_samples" | "created_at"
>;
type SortDir = "asc" | "desc";

/**
 * Clickable column-header button that displays a directional arrow when active.
 *
 * @param props.label - Column display label.
 * @param props.sortKey - The `Task` field this column sorts by.
 * @param props.current - The currently active sort key.
 * @param props.direction - The current sort direction.
 * @param props.onSort - Callback invoked when the header is clicked.
 */
function SortHeader({
  label,
  sortKey,
  current,
  direction,
  onSort,
}: {
  label: string;
  sortKey: SortKey;
  current: SortKey;
  direction: SortDir;
  onSort: (key: SortKey) => void;
}) {
  const isActive = current === sortKey;
  return (
    <button
      onClick={() => onSort(sortKey)}
      className="flex items-center gap-1 font-medium hover:text-foreground"
      data-testid={`sort-${sortKey}`}
    >
      {label}
      {isActive && (
        <span aria-hidden="true">{direction === "asc" ? "↑" : "↓"}</span>
      )}
    </button>
  );
}

/**
 * Modal dialog for submitting a new task to the OrcaMind embedding service.
 *
 * Validates required fields client-side before calling the submit callback.
 *
 * @param props.onClose - Callback invoked when the dialog is dismissed.
 * @param props.onSubmit - Callback invoked with the validated form data.
 * @param props.isSubmitting - Whether an embedding request is in flight.
 */
function EmbedTaskDialog({
  onClose,
  onSubmit,
  isSubmitting,
}: {
  onClose: () => void;
  onSubmit: (data: EmbedTaskRequest) => void;
  isSubmitting: boolean;
}) {
  const [form, setForm] = useState<EmbedTaskRequest>({
    name: "",
    domain: "",
    task_type: "",
    n_samples: 0,
  });
  const [errors, setErrors] = useState<Partial<Record<keyof EmbedTaskRequest, string>>>({});

  function validate(): boolean {
    const e: Partial<Record<keyof EmbedTaskRequest, string>> = {};
    if (!form.name.trim()) e.name = "Name is required";
    if (!form.domain.trim()) e.domain = "Domain is required";
    if (!form.task_type.trim()) e.task_type = "Task type is required";
    if (!form.n_samples || form.n_samples <= 0)
      e.n_samples = "Sample count must be a positive number";
    setErrors(e);
    return Object.keys(e).length === 0;
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (validate()) onSubmit(form);
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      data-testid="embed-dialog"
    >
      <div className="w-full max-w-lg rounded-lg bg-background p-6 shadow-lg">
        <h2 className="mb-4 text-xl font-semibold">Embed New Task</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <Input
            label="Name"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            error={errors.name}
            data-testid="embed-name"
          />
          <Input
            label="Domain"
            value={form.domain}
            onChange={(e) => setForm({ ...form, domain: e.target.value })}
            error={errors.domain}
            data-testid="embed-domain"
          />
          <Input
            label="Task Type"
            value={form.task_type}
            onChange={(e) => setForm({ ...form, task_type: e.target.value })}
            error={errors.task_type}
            data-testid="embed-task-type"
          />
          <Input
            label="Sample Count"
            type="number"
            value={form.n_samples || ""}
            onChange={(e) =>
              setForm({ ...form, n_samples: Number(e.target.value) })
            }
            error={errors.n_samples}
            data-testid="embed-n-samples"
          />
          <Input
            label="Feature Count (optional)"
            type="number"
            value={form.n_features ?? ""}
            onChange={(e) =>
              setForm({
                ...form,
                n_features: e.target.value ? Number(e.target.value) : undefined,
              })
            }
          />
          <Input
            label="Class Count (optional)"
            type="number"
            value={form.n_classes ?? ""}
            onChange={(e) =>
              setForm({
                ...form,
                n_classes: e.target.value ? Number(e.target.value) : undefined,
              })
            }
          />
          <div className="flex justify-end gap-3 pt-2">
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Embedding…" : "Embed Task"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}

/**
 * OrcaMind task list page.
 *
 * Displays all registered tasks in a table that supports client-side sorting
 * (by name, domain, type, sample count, or creation date) and text filtering
 * by name or domain. Clicking a row navigates to the task detail view.
 * The "Embed New Task" button opens a dialog for submitting a new task to the
 * OrcaMind embedding service, after which the list refreshes automatically.
 */
export function TaskList() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("name");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [dialogOpen, setDialogOpen] = useState(false);

  const {
    data: tasks = [],
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["orcamind-tasks"],
    queryFn: async () => {
      const res = await apiClient.get<Task[]>("/orcamind/tasks");
      return res.data;
    },
    staleTime: 30_000,
  });

  const { mutate: embedTask, isPending: isEmbedding } = useMutation({
    mutationFn: async (data: EmbedTaskRequest) => {
      const res = await apiClient.post<Task>("/orcamind/tasks", data);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["orcamind-tasks"] });
      setDialogOpen(false);
    },
  });

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  }

  const filteredSorted = useMemo(() => {
    const q = search.toLowerCase();
    const filtered = tasks.filter(
      (t) =>
        t.name.toLowerCase().includes(q) ||
        t.domain.toLowerCase().includes(q),
    );
    return [...filtered].sort((a, b) => {
      const av = a[sortKey] ?? "";
      const bv = b[sortKey] ?? "";
      const cmp = String(av).localeCompare(String(bv), undefined, {
        numeric: true,
      });
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [tasks, search, sortKey, sortDir]);

  return (
    <div>
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Tasks</h1>
        <Button
          onClick={() => setDialogOpen(true)}
          data-testid="embed-task-btn"
        >
          Embed New Task
        </Button>
      </div>
      <p className="mt-1 text-muted-foreground">
        Registered OrcaMind tasks and their metadata.
      </p>

      <div className="mt-4">
        <Input
          placeholder="Search by name or domain…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="max-w-sm"
          data-testid="task-search"
        />
      </div>

      <div className="mt-4 overflow-auto rounded-lg border">
        {isLoading ? (
          <p className="p-4 text-muted-foreground" data-testid="task-list-loading">
            Loading tasks…
          </p>
        ) : isError ? (
          <p className="p-4 text-destructive" data-testid="task-list-error">
            Failed to load tasks.
          </p>
        ) : filteredSorted.length === 0 ? (
          <p className="p-4 text-muted-foreground" data-testid="task-list-empty">
            No tasks found.
          </p>
        ) : (
          <table className="w-full text-sm" data-testid="task-table">
            <thead className="border-b bg-muted/50 text-left">
              <tr>
                <th className="px-4 py-3">
                  <SortHeader
                    label="Name"
                    sortKey="name"
                    current={sortKey}
                    direction={sortDir}
                    onSort={handleSort}
                  />
                </th>
                <th className="px-4 py-3">
                  <SortHeader
                    label="Domain"
                    sortKey="domain"
                    current={sortKey}
                    direction={sortDir}
                    onSort={handleSort}
                  />
                </th>
                <th className="px-4 py-3">
                  <SortHeader
                    label="Type"
                    sortKey="task_type"
                    current={sortKey}
                    direction={sortDir}
                    onSort={handleSort}
                  />
                </th>
                <th className="px-4 py-3">
                  <SortHeader
                    label="Samples"
                    sortKey="n_samples"
                    current={sortKey}
                    direction={sortDir}
                    onSort={handleSort}
                  />
                </th>
                <th className="px-4 py-3">
                  <SortHeader
                    label="Created"
                    sortKey="created_at"
                    current={sortKey}
                    direction={sortDir}
                    onSort={handleSort}
                  />
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredSorted.map((task) => (
                <tr
                  key={task.task_id}
                  className="cursor-pointer border-b last:border-0 hover:bg-muted/50"
                  tabIndex={0}
                  role="button"
                  onClick={() =>
                    navigate(`${ROUTES.ORCAMIND_TASKS}/${task.task_id}`)
                  }
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      navigate(`${ROUTES.ORCAMIND_TASKS}/${task.task_id}`);
                    }
                  }}
                  data-testid={`task-row-${task.task_id}`}
                >
                  <td className="px-4 py-3 font-medium">{task.name}</td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {task.domain}
                  </td>
                  <td className="px-4 py-3">{task.task_type}</td>
                  <td className="px-4 py-3">
                    {task.n_samples.toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {formatDate(task.created_at)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {dialogOpen && (
        <EmbedTaskDialog
          onClose={() => setDialogOpen(false)}
          onSubmit={embedTask}
          isSubmitting={isEmbedding}
        />
      )}
    </div>
  );
}
