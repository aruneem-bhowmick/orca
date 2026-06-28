import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { NotFound } from "@/components/NotFound";

describe("NotFound", () => {
  it("renders the 404 heading", () => {
    render(<NotFound />);
    expect(screen.getByTestId("not-found-page")).toBeInTheDocument();
    expect(screen.getByText("404")).toBeInTheDocument();
  });

  it("renders the 'Page not found' message", () => {
    render(<NotFound />);
    expect(screen.getByText("Page not found")).toBeInTheDocument();
  });

  it("renders a Go Home link pointing to '/'", () => {
    render(<NotFound />);
    const link = screen.getByTestId("not-found-go-home");
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute("href", "/");
  });
});
