import { describe, it, expect } from "vitest";
import { ROUTES, NAV_ITEMS } from "@/lib/constants";
import type { NavItem } from "@/lib/constants";

describe("ROUTES", () => {
  it("defines all public routes", () => {
    expect(ROUTES.HOME).toBe("/");
    expect(ROUTES.LOGIN).toBe("/login");
    expect(ROUTES.REGISTER).toBe("/register");
    expect(ROUTES.OAUTH_CALLBACK).toBe("/oauth/callback");
  });

  it("defines the dashboard route", () => {
    expect(ROUTES.DASHBOARD).toBe("/dashboard");
  });

  it("defines OrcaMind routes under /dashboard/orcamind", () => {
    expect(ROUTES.ORCAMIND_TASKS).toBe("/dashboard/orcamind/tasks");
    expect(ROUTES.ORCAMIND_TASK_DETAIL).toBe("/dashboard/orcamind/tasks/:id");
  });

  it("defines OrcaLab routes under /dashboard/orcalab", () => {
    expect(ROUTES.ORCALAB_EXPERIMENTS).toBe("/dashboard/orcalab/experiments");
    expect(ROUTES.ORCALAB_EXPERIMENT_DETAIL).toBe(
      "/dashboard/orcalab/experiments/:id",
    );
    expect(ROUTES.ORCALAB_SWEEPS).toBe("/dashboard/orcalab/sweeps");
  });

  it("defines OrcaNet routes under /dashboard/orcanet", () => {
    expect(ROUTES.ORCANET_TRANSFER).toBe("/dashboard/orcanet/transfer");
    expect(ROUTES.ORCANET_RETRIEVE).toBe("/dashboard/orcanet/retrieve");
  });

  it("defines history sub-routes", () => {
    expect(ROUTES.HISTORY).toBe("/history");
    expect(ROUTES.HISTORY_TASKS).toBe("/history/tasks");
    expect(ROUTES.HISTORY_EXPERIMENTS).toBe("/history/experiments");
  });

  it("defines bookmarks and profile routes", () => {
    expect(ROUTES.BOOKMARKS).toBe("/bookmarks");
    expect(ROUTES.PROFILE).toBe("/profile");
  });
});

describe("NAV_ITEMS", () => {
  it("contains six top-level items", () => {
    expect(NAV_ITEMS).toHaveLength(6);
  });

  it("has Dashboard as the first item without children", () => {
    const dashboard = NAV_ITEMS[0];
    expect(dashboard.label).toBe("Dashboard");
    expect(dashboard.children).toBeUndefined();
  });

  it("has OrcaMind as a group with Tasks sub-item", () => {
    const orcaMind = NAV_ITEMS[1];
    expect(orcaMind.label).toBe("OrcaMind");
    expect(orcaMind.children).toBeDefined();
    expect(orcaMind.children!.length).toBe(1);
    expect(orcaMind.children![0].label).toBe("Tasks");
  });

  it("has OrcaLab as a group with Experiments and Sweeps sub-items", () => {
    const orcaLab = NAV_ITEMS[2];
    expect(orcaLab.label).toBe("OrcaLab");
    expect(orcaLab.children).toBeDefined();
    expect(orcaLab.children!.length).toBe(2);
    const childLabels = orcaLab.children!.map((c: NavItem) => c.label);
    expect(childLabels).toEqual(["Experiments", "Sweeps"]);
  });

  it("has OrcaNet as a group with Transfer and Retrieval sub-items", () => {
    const orcaNet = NAV_ITEMS[3];
    expect(orcaNet.label).toBe("OrcaNet");
    expect(orcaNet.children).toBeDefined();
    expect(orcaNet.children!.length).toBe(2);
    const childLabels = orcaNet.children!.map((c: NavItem) => c.label);
    expect(childLabels).toEqual(["Transfer", "Retrieval"]);
  });

  it("has History and Bookmarks as flat items", () => {
    const history = NAV_ITEMS[4];
    const bookmarks = NAV_ITEMS[5];
    expect(history.label).toBe("History");
    expect(history.children).toBeUndefined();
    expect(bookmarks.label).toBe("Bookmarks");
    expect(bookmarks.children).toBeUndefined();
  });

  it("every item has an icon defined", () => {
    function checkIcon(item: NavItem) {
      expect(item.icon).toBeTruthy();
      item.children?.forEach(checkIcon);
    }
    NAV_ITEMS.forEach(checkIcon);
  });
});
