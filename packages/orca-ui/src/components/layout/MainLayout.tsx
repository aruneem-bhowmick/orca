import { useState } from "react";
import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { Header } from "./Header";

/**
 * Main application layout wrapping all protected routes.
 *
 * Composes the collapsible sidebar, top header bar, and a scrollable content
 * area that renders the matched child route via React Router's `<Outlet />`.
 * The layout fills the full viewport height.
 *
 * On desktop (md breakpoint and above) the sidebar is permanently visible on
 * the left. On mobile the sidebar is hidden by default; pressing the hamburger
 * button in the header opens it as a slide-in drawer with a semi-transparent
 * backdrop overlay. Tapping the backdrop or any navigation link closes the
 * drawer automatically.
 */
export function MainLayout() {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        mobileOpen={mobileOpen}
        onMobileClose={() => setMobileOpen(false)}
      />
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header onMenuToggle={() => setMobileOpen(true)} />
        <main className="flex-1 overflow-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
