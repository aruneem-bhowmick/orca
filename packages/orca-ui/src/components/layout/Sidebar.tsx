import { useState } from "react";
import { NavLink } from "react-router-dom";
import { cn } from "@/lib/utils";
import { ROUTES } from "@/lib/constants";
import { useAuth } from "@/hooks/useAuth";

const navItems = [
  { label: "Dashboard", path: ROUTES.DASHBOARD, icon: "grid" },
  { label: "Tasks", path: ROUTES.TASKS, icon: "cpu" },
  { label: "Experiments", path: ROUTES.EXPERIMENTS, icon: "flask" },
  { label: "Sweeps", path: ROUTES.SWEEPS, icon: "search" },
  { label: "Transfers", path: ROUTES.TRANSFERS, icon: "share" },
  { label: "History", path: ROUTES.HISTORY, icon: "clock" },
  { label: "Bookmarks", path: ROUTES.BOOKMARKS, icon: "bookmark" },
];

const iconMap: Record<string, string> = {
  grid: "M4 4h7v7H4V4zm9 0h7v7h-7V4zm-9 9h7v7H4v-7zm9 0h7v7h-7v-7z",
  cpu: "M9 3v2H6a1 1 0 0 0-1 1v3H3v2h2v4H3v2h2v3a1 1 0 0 0 1 1h3v2h2v-2h4v2h2v-2h3a1 1 0 0 0 1-1v-3h2v-2h-2v-4h2V9h-2V6a1 1 0 0 0-1-1h-3V3h-2v2h-4V3H9z",
  flask: "M10 2v6.292a4 4 0 0 0-1.17 1.17L4 16.586V20a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3.414l-4.83-7.124A4 4 0 0 0 14 8.292V2h-4z",
  search: "M10 2a8 8 0 1 0 5.293 14.293l4.707 4.707 1.414-1.414-4.707-4.707A8 8 0 0 0 10 2zm0 2a6 6 0 1 1 0 12 6 6 0 0 1 0-12z",
  share: "M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05-4.11A2.99 2.99 0 0 0 18 8a3 3 0 1 0-3-3c0 .24.04.47.09.7L8.04 9.81A2.99 2.99 0 0 0 6 9a3 3 0 0 0 0 6c.79 0 1.5-.31 2.04-.81l7.12 4.16c-.05.21-.08.43-.08.65a2.92 2.92 0 0 0 2.92 2.92A2.92 2.92 0 0 0 21 19a2.92 2.92 0 0 0-3-2.92z",
  clock: "M12 2C6.486 2 2 6.486 2 12s4.486 10 10 10 10-4.486 10-10S17.514 2 12 2zm0 18c-4.411 0-8-3.589-8-8s3.589-8 8-8 8 3.589 8 8-3.589 8-8 8zm1-13h-2v6l5.25 3.15.75-1.23-4-2.42V7z",
  bookmark: "M5 3v18l7-5 7 5V3H5z",
};

function NavIcon({ icon, collapsed }: { icon: string; collapsed: boolean }) {
  return (
    <svg
      className={cn("shrink-0", collapsed ? "h-6 w-6" : "h-5 w-5")}
      viewBox="0 0 24 24"
      fill="currentColor"
    >
      <path d={iconMap[icon] || iconMap.grid} />
    </svg>
  );
}

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const { user, logout } = useAuth();

  return (
    <aside
      className={cn(
        "flex h-screen flex-col border-r bg-card transition-all duration-200",
        collapsed ? "w-16" : "w-60",
      )}
      data-testid="sidebar"
    >
      <div className="flex items-center justify-between border-b p-4">
        {!collapsed && <span className="text-lg font-bold">Orca</span>}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="rounded p-1 hover:bg-accent"
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          data-testid="sidebar-toggle"
        >
          <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d={collapsed ? "M13 5l7 7-7 7M5 5l7 7-7 7" : "M11 19l-7-7 7-7M19 19l-7-7 7-7"}
            />
          </svg>
        </button>
      </div>

      <nav className="flex-1 space-y-1 p-2">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            aria-label={item.label}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                collapsed && "justify-center px-2",
              )
            }
          >
            <NavIcon icon={item.icon} collapsed={collapsed} />
            {!collapsed && <span>{item.label}</span>}
          </NavLink>
        ))}
      </nav>

      <div className="border-t p-4">
        <div className={cn("flex items-center gap-3", collapsed && "justify-center")}>
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-sm font-medium text-primary-foreground">
            {user?.username?.charAt(0).toUpperCase() || "U"}
          </div>
          {!collapsed && (
            <div className="flex-1 overflow-hidden">
              <p className="truncate text-sm font-medium">{user?.username}</p>
              <button
                onClick={logout}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                Sign out
              </button>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
