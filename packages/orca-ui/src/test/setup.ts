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
// The theme store calls window.matchMedia at initialization to detect system
// color scheme preference. jsdom does not implement matchMedia, so stub it.
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }),
});

global.IntersectionObserver = class IntersectionObserver {
  static observers: IntersectionObserver[] = [];
  
  readonly root = null;
  readonly rootMargin = "";
  readonly thresholds: readonly number[] = [];
  
  callback: IntersectionObserverCallback;
  elements: Element[] = [];

  constructor(callback: IntersectionObserverCallback) {
    this.callback = callback;
    IntersectionObserver.observers.push(this);
  }

  observe(element: Element) {
    this.elements.push(element);
  }

  unobserve(element: Element) {
    this.elements = this.elements.filter((el) => el !== element);
  }

  disconnect() {
    this.elements = [];
  }

  takeRecords(): IntersectionObserverEntry[] {
    return [];
  }

  trigger(isIntersecting: boolean, targetElement: Element) {
    const entry: IntersectionObserverEntry = {
      isIntersecting,
      target: targetElement,
      time: Date.now(),
      intersectionRatio: isIntersecting ? 1 : 0,
      boundingClientRect: {} as DOMRectReadOnly,
      intersectionRect: {} as DOMRectReadOnly,
      rootBounds: null,
    };
    this.callback([entry], this);
  }
};
