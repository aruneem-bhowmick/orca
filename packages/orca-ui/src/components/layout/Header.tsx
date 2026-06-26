import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { useAuth } from "@/hooks/useAuth";
import { ROUTE_LABELS } from "@/lib/constants";

/**
 * Build breadcrumb segments from the current URL pathname.
 *
 * Splits the pathname on `/`, filters empty segments, and produces
 * an array of `{ label, path }` objects. Labels are derived by
 * capitalising the segment and replacing known prefixes with
 * human-readable names.
 *
 * @param pathname - The current `location.pathname`.
 * @returns Array of breadcrumb items with label and full path.
 */
function buildBreadcrumbs(pathname: string) {
  const segments = pathname.split("/").filter(Boolean);
  const crumbs: { label: string; path: string }[] = [];

  let accumulated = "";
  for (const segment of segments) {
    accumulated += `/${segment}`;
    const label =
      ROUTE_LABELS[segment] ||
      segment.charAt(0).toUpperCase() + segment.slice(1);
    crumbs.push({ label, path: accumulated });
  }

  return crumbs;
}

/**
 * Top header bar for the main application layout.
 *
 * Displays a breadcrumb trail showing the current navigation context,
 * a search input placeholder for future global search, a notifications
 * bell icon with a badge count, and a dark mode toggle. The dark mode
 * preference is persisted to localStorage and applied via the `dark`
 * class on the document root element.
 */
export function Header() {
  const { user } = useAuth();
  const location = useLocation();
  const breadcrumbs = buildBreadcrumbs(location.pathname);

  const [darkMode, setDarkMode] = useState(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("darkMode");
      if (saved !== null) {
        return saved === "true";
      }
      return document.documentElement.classList.contains("dark");
    }
    return false;
  });

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
    localStorage.setItem("darkMode", String(darkMode));
  }, [darkMode]);

  return (
    <header
      className="flex h-14 items-center justify-between border-b bg-card px-6"
      data-testid="header"
    >
      {/* Breadcrumb navigation */}
      <nav className="flex items-center gap-1 text-sm" aria-label="Breadcrumb" data-testid="breadcrumbs">
        {breadcrumbs.map((crumb, index) => (
          <span key={crumb.path} className="flex items-center gap-1">
            {index > 0 && (
              <span className="text-muted-foreground" aria-hidden="true">
                /
              </span>
            )}
            {index === breadcrumbs.length - 1 ? (
              <span className="font-medium text-foreground">{crumb.label}</span>
            ) : (
              <Link
                to={crumb.path}
                className="text-muted-foreground hover:text-foreground"
              >
                {crumb.label}
              </Link>
            )}
          </span>
        ))}
        {breadcrumbs.length === 0 && (
          <span className="font-medium text-foreground">Home</span>
        )}
      </nav>

      <div className="flex items-center gap-4">
        {/* Search input placeholder */}
        <div className="relative hidden md:block" data-testid="search-container">
          <svg
            className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
          <input
            type="text"
            placeholder="Search..."
            className="h-9 w-64 rounded-md border bg-background pl-9 pr-4 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary"
            aria-label="Search"
            data-testid="search-input"
            readOnly
          />
        </div>

        {/* Notifications bell */}
        <button
          className="relative rounded-md p-2 hover:bg-accent"
          aria-label="Notifications"
          data-testid="notifications-button"
        >
          <svg
            className="h-5 w-5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
            />
          </svg>
          <span
            className="absolute right-1 top-1 flex h-4 w-4 items-center justify-center rounded-full bg-destructive text-[10px] font-bold text-white"
            data-testid="notification-badge"
          >
            0
          </span>
        </button>

        {/* Dark mode toggle */}
        <button
          onClick={() => setDarkMode(!darkMode)}
          className="rounded-md p-2 hover:bg-accent"
          aria-label="Toggle dark mode"
          data-testid="dark-mode-toggle"
        >
          {darkMode ? (
            <svg className="h-5 w-5" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 3a9 9 0 1 0 9 9c0-.46-.04-.92-.1-1.36a5.389 5.389 0 0 1-4.4 2.26 5.403 5.403 0 0 1-3.14-9.8c-.44-.06-.9-.1-1.36-.1z" />
            </svg>
          ) : (
            <svg className="h-5 w-5" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 7c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5zM2 13h2c.55 0 1-.45 1-1s-.45-1-1-1H2c-.55 0-1 .45-1 1s.45 1 1 1zm18 0h2c.55 0 1-.45 1-1s-.45-1-1-1h-2c-.55 0-1 .45-1 1s.45 1 1 1zM11 2v2c0 .55.45 1 1 1s1-.45 1-1V2c0-.55-.45-1-1-1s-1 .45-1 1zm0 18v2c0 .55.45 1 1 1s1-.45 1-1v-2c0-.55-.45-1-1-1s-1 .45-1 1zM5.99 4.58a.996.996 0 0 0-1.41 0 .996.996 0 0 0 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0s.39-1.03 0-1.41L5.99 4.58zm12.37 12.37a.996.996 0 0 0-1.41 0 .996.996 0 0 0 0 1.41l1.06 1.06c.39.39 1.03.39 1.41 0a.996.996 0 0 0 0-1.41l-1.06-1.06zm1.06-10.96a.996.996 0 0 0 0-1.41.996.996 0 0 0-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41s1.03.39 1.41 0l1.06-1.06zM7.05 18.36a.996.996 0 0 0 0-1.41.996.996 0 0 0-1.41 0l-1.06 1.06c-.39.39-.39 1.03 0 1.41s1.03.39 1.41 0l1.06-1.06z" />
            </svg>
          )}
        </button>

        {/* User info */}
        {user && (
          <div className="flex items-center gap-2" data-testid="user-menu">
            <span className="text-sm text-muted-foreground">{user.email}</span>
          </div>
        )}
      </div>
    </header>
  );
}
