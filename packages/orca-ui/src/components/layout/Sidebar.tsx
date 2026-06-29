import { useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { ROUTES, NAV_ITEMS } from "@/lib/constants";
import type { NavItem } from "@/lib/constants";
import { useAuth } from "@/hooks/useAuth";

/**
 * SVG path data for sidebar navigation icons, keyed by icon identifier.
 * Each path is designed for a 24x24 viewBox.
 */
const iconMap: Record<string, string> = {
  grid: "M4 4h7v7H4V4zm9 0h7v7h-7V4zm-9 9h7v7H4v-7zm9 0h7v7h-7v-7z",
  cpu: "M9 3v2H6a1 1 0 0 0-1 1v3H3v2h2v4H3v2h2v3a1 1 0 0 0 1 1h3v2h2v-2h4v2h2v-2h3a1 1 0 0 0 1-1v-3h2v-2h-2v-4h2V9h-2V6a1 1 0 0 0-1-1h-3V3h-2v2h-4V3H9z",
  flask:
    "M10 2v6.292a4 4 0 0 0-1.17 1.17L4 16.586V20a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3.414l-4.83-7.124A4 4 0 0 0 14 8.292V2h-4z",
  search:
    "M10 2a8 8 0 1 0 5.293 14.293l4.707 4.707 1.414-1.414-4.707-4.707A8 8 0 0 0 10 2zm0 2a6 6 0 1 1 0 12 6 6 0 0 1 0-12z",
  share:
    "M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05-4.11A2.99 2.99 0 0 0 18 8a3 3 0 1 0-3-3c0 .24.04.47.09.7L8.04 9.81A2.99 2.99 0 0 0 6 9a3 3 0 0 0 0 6c.79 0 1.5-.31 2.04-.81l7.12 4.16c-.05.21-.08.43-.08.65a2.92 2.92 0 0 0 2.92 2.92A2.92 2.92 0 0 0 21 19a2.92 2.92 0 0 0-3-2.92z",
  clock:
    "M12 2C6.486 2 2 6.486 2 12s4.486 10 10 10 10-4.486 10-10S17.514 2 12 2zm0 18c-4.411 0-8-3.589-8-8s3.589-8 8-8 8 3.589 8 8-3.589 8-8 8zm1-13h-2v6l5.25 3.15.75-1.23-4-2.42V7z",
  bookmark: "M5 3v18l7-5 7 5V3H5z",
  settings:
    "M19.14 12.94c.04-.31.06-.63.06-.94 0-.31-.02-.63-.06-.94l2.03-1.58a.49.49 0 0 0 .12-.61l-1.92-3.32a.49.49 0 0 0-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54a.484.484 0 0 0-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96a.49.49 0 0 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.04.31-.06.63-.06.94s.02.63.06.94l-2.03 1.58a.49.49 0 0 0-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6A3.6 3.6 0 1 1 12 8.4a3.6 3.6 0 0 1 0 7.2z",
  logout:
    "M17 7l-1.41 1.41L18.17 11H8v2h10.17l-2.58 2.59L17 17l5-5-5-5zM4 5h8V3H4c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h8v-2H4V5z",
  user: "M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z",
  close: "M6 18L18 6M6 6l12 12",
};

/**
 * Render an SVG icon from the icon map.
 *
 * @param props.icon - Key into `iconMap` for the SVG path data.
 * @param props.collapsed - Whether the sidebar is in collapsed mode (renders larger icons).
 */
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

/**
 * Render a single navigation link with icon and optional label.
 *
 * @param props.item - Navigation item definition.
 * @param props.collapsed - Whether the sidebar is collapsed (hides labels).
 * @param props.indented - Whether this item is a child of a group (adds left padding).
 * @param props.onNavigate - Optional callback invoked after navigation (used
 *   on mobile to close the drawer after the user taps a link).
 */
function SidebarLink({
  item,
  collapsed,
  indented = false,
  onNavigate,
}: {
  item: NavItem;
  collapsed: boolean;
  indented?: boolean;
  onNavigate?: () => void;
}) {
  return (
    <NavLink
      key={item.path}
      to={item.path}
      end={item.path === ROUTES.DASHBOARD}
      aria-label={item.label}
      onClick={onNavigate}
      className={({ isActive }) =>
        cn(
          "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
          isActive
            ? "bg-primary text-primary-foreground"
            : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
          collapsed && "justify-center px-2",
          indented && !collapsed && "pl-9",
        )
      }
    >
      <NavIcon icon={item.icon} collapsed={collapsed} />
      {!collapsed && <span>{item.label}</span>}
    </NavLink>
  );
}

/**
 * Render a navigation group with expandable sub-items.
 *
 * When the sidebar is collapsed, only the parent icon is shown.
 * When expanded, the parent label is clickable to toggle the
 * sub-item list, and child items are rendered indented below.
 *
 * @param props.item - Navigation group with children.
 * @param props.collapsed - Whether the sidebar is collapsed.
 * @param props.onNavigate - Callback forwarded to child `SidebarLink`s.
 */
function SidebarGroup({
  item,
  collapsed,
  onNavigate,
}: {
  item: NavItem;
  collapsed: boolean;
  onNavigate?: () => void;
}) {
  const location = useLocation();
  const isChildActive = item.children?.some((child) =>
    location.pathname.startsWith(child.path),
  );
  const [expanded, setExpanded] = useState(isChildActive ?? false);

  // Keep the group expanded when navigating to one of its children.
  useEffect(() => {
    if (isChildActive) {
      setExpanded(true);
    }
  }, [isChildActive]);

  if (collapsed) {
    return (
      <NavLink
        to={item.path}
        aria-label={item.label}
        className={({ isActive }) =>
          cn(
            "flex items-center justify-center rounded-md px-2 py-2 text-sm font-medium transition-colors",
            isActive || isChildActive
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
          )
        }
      >
        <NavIcon icon={item.icon} collapsed={collapsed} />
      </NavLink>
    );
  }

  return (
    <div>
      <button
        onClick={() => setExpanded(!expanded)}
        className={cn(
          "flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
          isChildActive
            ? "text-foreground"
            : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
        )}
        aria-label={item.label}
        data-testid={`nav-group-${item.label.toLowerCase()}`}
      >
        <NavIcon icon={item.icon} collapsed={collapsed} />
        <span className="flex-1 text-left">{item.label}</span>
        <svg
          className={cn(
            "h-4 w-4 transition-transform",
            expanded && "rotate-90",
          )}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 5l7 7-7 7"
          />
        </svg>
      </button>
      {expanded && (
        <div className="mt-1 space-y-1" data-testid={`nav-children-${item.label.toLowerCase()}`}>
          {item.children!.map((child) => (
            <SidebarLink
              key={child.path}
              item={child}
              collapsed={collapsed}
              indented
              onNavigate={onNavigate}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Collapsible sidebar navigation for the main application layout.
 *
 * On desktop (md breakpoint and above) the sidebar is permanently visible and
 * can be collapsed to icon-only mode (64px) via the collapse toggle button.
 * On mobile the sidebar is hidden by default; it slides in as a full-height
 * drawer when `mobileOpen` is `true`. Tapping a navigation link or the
 * backdrop calls `onMobileClose` to close the drawer.
 *
 * @param props.mobileOpen - Whether the mobile drawer is visible.
 * @param props.onMobileClose - Callback invoked to close the mobile drawer.
 */
export function Sidebar({
  mobileOpen = false,
  onMobileClose,
}: {
  mobileOpen?: boolean;
  onMobileClose?: () => void;
}) {
  const [collapsed, setCollapsed] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const { user, logout } = useAuth();

  const sidebarContent = (isMobile: boolean) => (
    <aside
      className={cn(
        "flex h-screen flex-col border-r bg-card transition-all duration-200",
        isMobile ? "w-72" : collapsed ? "w-16" : "w-60",
      )}
      data-testid={isMobile ? "sidebar-mobile" : "sidebar"}
    >
      {/* Logo and collapse toggle */}
      <div className="flex items-center justify-between border-b p-4">
        {(!collapsed || isMobile) && (
          <span className="text-lg font-bold">Orca</span>
        )}
        {isMobile ? (
          <button
            onClick={onMobileClose}
            className="rounded p-1 hover:bg-accent"
            aria-label="Close navigation menu"
            data-testid="sidebar-mobile-close"
          >
            <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={iconMap.close} />
            </svg>
          </button>
        ) : (
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="rounded p-1 hover:bg-accent"
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            data-testid="sidebar-toggle"
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
                d={
                  collapsed
                    ? "M13 5l7 7-7 7M5 5l7 7-7 7"
                    : "M11 19l-7-7 7-7M19 19l-7-7 7-7"
                }
              />
            </svg>
          </button>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 overflow-y-auto p-2">
        {NAV_ITEMS.map((item) =>
          item.children ? (
            <SidebarGroup
              key={item.label}
              item={item}
              collapsed={!isMobile && collapsed}
              onNavigate={isMobile ? onMobileClose : undefined}
            />
          ) : (
            <SidebarLink
              key={item.path}
              item={item}
              collapsed={!isMobile && collapsed}
              onNavigate={isMobile ? onMobileClose : undefined}
            />
          ),
        )}
      </nav>

      {/* User section with dropdown */}
      <div className="relative border-t p-4">
        <button
          onClick={() => setUserMenuOpen(!userMenuOpen)}
          className={cn(
            "flex w-full items-center gap-3 rounded-md p-1 hover:bg-accent",
            !isMobile && collapsed && "justify-center",
          )}
          data-testid="user-menu-trigger"
        >
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-sm font-medium text-primary-foreground">
            {user?.username?.charAt(0).toUpperCase() || "U"}
          </div>
          {(!collapsed || isMobile) && (
            <div className="flex-1 overflow-hidden text-left">
              <p className="truncate text-sm font-medium">{user?.username}</p>
              <p className="truncate text-xs text-muted-foreground">
                {user?.email}
              </p>
            </div>
          )}
        </button>

        {/* Dropdown menu */}
        {userMenuOpen && (
          <div
            className={cn(
              "absolute bottom-full mb-2 w-48 rounded-md border bg-card py-1 shadow-lg",
              !isMobile && collapsed ? "left-16" : "left-4",
            )}
            data-testid="user-dropdown"
          >
            <NavLink
              to={ROUTES.PROFILE}
              className="flex items-center gap-2 px-4 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-foreground"
              onClick={() => {
                setUserMenuOpen(false);
                if (isMobile) onMobileClose?.();
              }}
            >
              <NavIcon icon="user" collapsed={false} />
              Profile
            </NavLink>
            <button
              onClick={() => {
                setUserMenuOpen(false);
                logout();
              }}
              className="flex w-full items-center gap-2 px-4 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-foreground"
              data-testid="logout-button"
            >
              <NavIcon icon="logout" collapsed={false} />
              Sign out
            </button>
          </div>
        )}
      </div>
    </aside>
  );

  return (
    <>
      {/* Desktop sidebar — hidden on mobile */}
      <div className="hidden md:flex">
        {sidebarContent(false)}
      </div>

      {/* Mobile drawer overlay */}
      {mobileOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-40 bg-black/50 md:hidden"
            onClick={onMobileClose}
            aria-hidden="true"
            data-testid="sidebar-backdrop"
          />
          {/* Drawer */}
          <div className="fixed inset-y-0 left-0 z-50 flex md:hidden">
            {sidebarContent(true)}
          </div>
        </>
      )}
    </>
  );
}
