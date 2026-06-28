import "@testing-library/jest-dom";

// Recharts ResponsiveContainer relies on ResizeObserver, which jsdom does not
// implement. Provide a no-op stub so tests that render charts don't throw.
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};

// The ActivityLog page uses IntersectionObserver to trigger infinite-scroll
// page loads when a sentinel element enters the viewport. jsdom does not
// implement IntersectionObserver, so provide a no-op stub here.
global.IntersectionObserver = class IntersectionObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
  readonly root = null;
  readonly rootMargin = "";
  readonly thresholds: readonly number[] = [];
  takeRecords(): IntersectionObserverEntry[] { return []; }
};
