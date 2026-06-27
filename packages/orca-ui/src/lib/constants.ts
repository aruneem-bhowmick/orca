/**
 * Application-wide constants.
 *
 * @module lib/constants
 */

/**
 * Base URL for all BFF API requests. Reads from the `VITE_API_BASE_URL`
 * environment variable at build time, falling back to `"/api/v1"` which
 * is handled by Vite's dev proxy or the production nginx config.
 */
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api/v1";

/**
 * All named route paths in the application.
 *
 * Public routes: HOME, LOGIN, REGISTER, OAUTH_CALLBACK.
 * Protected routes (require authentication): all others.
 *
 * Dashboard sub-routes are nested under `/dashboard` and organised
 * by service (orcamind, orcalab, orcanet) with detail-view patterns
 * using `:id` parameters.
 */
export const ROUTES = {
  HOME: "/",
  LOGIN: "/login",
  REGISTER: "/register",
  OAUTH_CALLBACK: "/oauth/callback",
  DASHBOARD: "/dashboard",
  ORCAMIND_TASKS: "/dashboard/orcamind/tasks",
  ORCAMIND_TASK_DETAIL: "/dashboard/orcamind/tasks/:id",
  ORCALAB_EXPERIMENTS: "/dashboard/orcalab/experiments",
  ORCALAB_EXPERIMENT_DETAIL: "/dashboard/orcalab/experiments/:id",
  ORCALAB_SWEEPS: "/dashboard/orcalab/sweeps",
  ORCANET_TRANSFER: "/dashboard/orcanet/transfer",
  ORCANET_RETRIEVE: "/dashboard/orcanet/retrieve",
  ORCAMIND_RECOMMENDATIONS: "/dashboard/orcamind/recommendations",
  HISTORY: "/history",
  HISTORY_TASKS: "/history/tasks",
  HISTORY_EXPERIMENTS: "/history/experiments",
  BOOKMARKS: "/bookmarks",
  PROFILE: "/profile",
} as const;

/** Union of all route path literals defined in `ROUTES`. */
export type RoutePath = (typeof ROUTES)[keyof typeof ROUTES];

/**
 * Human-readable labels for URL path segments, used by the
 * breadcrumb builder to convert slugs like "orcamind" into
 * display names like "OrcaMind".
 */
export const ROUTE_LABELS: Record<string, string> = {
  dashboard: "Dashboard",
  orcamind: "OrcaMind",
  orcalab: "OrcaLab",
  orcanet: "OrcaNet",
  tasks: "Tasks",
  recommendations: "Recommendations",
  experiments: "Experiments",
  sweeps: "Sweeps",
  transfer: "Transfer",
  retrieve: "Retrieval",
  history: "History",
  bookmarks: "Bookmarks",
  profile: "Profile",
};

/**
 * Navigation items for the sidebar, organised by service group.
 *
 * Top-level items render as direct links. Items with `children`
 * render as expandable groups with indented sub-links.
 */
export interface NavItem {
  /** Display label in the sidebar. */
  label: string;
  /** Route path — must match a value in `ROUTES`. */
  path: RoutePath;
  /** SVG icon identifier used by the sidebar icon map. */
  icon: string;
  /** Optional nested sub-items shown when the group is expanded. */
  children?: NavItem[];
}

/** Sidebar navigation structure with grouped service sub-items. */
export const NAV_ITEMS: NavItem[] = [
  { label: "Dashboard", path: ROUTES.DASHBOARD, icon: "grid" },
  {
    label: "OrcaMind",
    path: ROUTES.ORCAMIND_TASKS,
    icon: "cpu",
    children: [
      { label: "Tasks", path: ROUTES.ORCAMIND_TASKS, icon: "cpu" },
      {
        label: "Recommendations",
        path: ROUTES.ORCAMIND_RECOMMENDATIONS,
        icon: "search",
      },
    ],
  },
  {
    label: "OrcaLab",
    path: ROUTES.ORCALAB_EXPERIMENTS,
    icon: "flask",
    children: [
      { label: "Experiments", path: ROUTES.ORCALAB_EXPERIMENTS, icon: "flask" },
      { label: "Sweeps", path: ROUTES.ORCALAB_SWEEPS, icon: "search" },
    ],
  },
  {
    label: "OrcaNet",
    path: ROUTES.ORCANET_TRANSFER,
    icon: "share",
    children: [
      { label: "Transfer", path: ROUTES.ORCANET_TRANSFER, icon: "share" },
      { label: "Retrieval", path: ROUTES.ORCANET_RETRIEVE, icon: "search" },
    ],
  },
  { label: "History", path: ROUTES.HISTORY, icon: "clock" },
  { label: "Bookmarks", path: ROUTES.BOOKMARKS, icon: "bookmark" },
];
