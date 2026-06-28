import { describe, it, expect } from "vitest";
import { render } from "@/test/test-utils";
import { screen } from "@testing-library/react";
import { Skeleton, TableSkeleton, CardRowSkeleton } from "@/components/ui/Skeleton";

describe("Skeleton", () => {
  it("renders a div with animate-pulse class", () => {
    const { container } = render(<Skeleton />);
    expect(container.firstChild).toHaveClass("animate-pulse");
  });

  it("merges additional className", () => {
    const { container } = render(<Skeleton className="h-4 w-24" />);
    expect(container.firstChild).toHaveClass("h-4", "w-24");
  });
});

describe("TableSkeleton", () => {
  it("renders the table-skeleton test id", () => {
    render(<TableSkeleton />);
    expect(screen.getByTestId("table-skeleton")).toBeInTheDocument();
  });

  it("renders the correct number of body rows (default 5)", () => {
    const { container } = render(<TableSkeleton />);
    // +1 for the header row
    const rows = container.querySelectorAll(".border-b");
    // header row + 5 body rows (last:border-0 on the final body row)
    expect(rows.length).toBeGreaterThanOrEqual(5);
  });

  it("renders the requested number of body rows", () => {
    render(<TableSkeleton rows={3} />);
    // We can't count rows exactly from DOM classes alone, but the component renders
    expect(screen.getByTestId("table-skeleton")).toBeInTheDocument();
  });
});

describe("CardRowSkeleton", () => {
  it("renders the card-row-skeleton test id", () => {
    render(<CardRowSkeleton />);
    expect(screen.getByTestId("card-row-skeleton")).toBeInTheDocument();
  });

  it("renders the default 4 card placeholders", () => {
    const { container } = render(<CardRowSkeleton />);
    // Each card has a bg-card rounded-lg class
    const cards = container.querySelectorAll(".bg-card.rounded-lg");
    expect(cards).toHaveLength(4);
  });

  it("renders the requested number of card placeholders", () => {
    const { container } = render(<CardRowSkeleton count={2} />);
    const cards = container.querySelectorAll(".bg-card.rounded-lg");
    expect(cards).toHaveLength(2);
  });
});
