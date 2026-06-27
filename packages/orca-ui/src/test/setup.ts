import "@testing-library/jest-dom";

// Recharts ResponsiveContainer relies on ResizeObserver, which jsdom does not
// implement. Provide a no-op stub so tests that render charts don't throw.
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};
