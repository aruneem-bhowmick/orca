import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { Header } from "./Header";

/**
 * Main application layout wrapping all protected routes.
 *
 * Composes the collapsible sidebar, top header bar, and a scrollable
 * content area that renders the matched child route via React Router's
 * `<Outlet />`. The layout fills the full viewport height.
 */
export function MainLayout() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
