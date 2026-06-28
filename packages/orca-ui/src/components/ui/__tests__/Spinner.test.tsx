import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Spinner, SpinnerOverlay } from "@/components/ui/Spinner";

describe("Spinner", () => {
  it("renders an svg with role=status", () => {
    render(<Spinner />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("uses the default aria-label 'Loading'", () => {
    render(<Spinner />);
    expect(screen.getByLabelText("Loading")).toBeInTheDocument();
  });

  it("accepts a custom label", () => {
    render(<Spinner label="Saving…" />);
    expect(screen.getByLabelText("Saving…")).toBeInTheDocument();
  });

  it("applies the animate-spin class", () => {
    render(<Spinner />);
    expect(screen.getByRole("status")).toHaveClass("animate-spin");
  });

  it("accepts a custom className", () => {
    render(<Spinner className="h-10 w-10" />);
    expect(screen.getByRole("status")).toHaveClass("h-10", "w-10");
  });
});

describe("SpinnerOverlay", () => {
  it("renders the overlay container", () => {
    render(<SpinnerOverlay />);
    expect(screen.getByTestId("spinner-overlay")).toBeInTheDocument();
  });

  it("contains an inner spinner with the default label", () => {
    render(<SpinnerOverlay />);
    expect(screen.getByLabelText("Loading")).toBeInTheDocument();
  });

  it("forwards a custom label to the inner Spinner", () => {
    render(<SpinnerOverlay label="Submitting" />);
    expect(screen.getByLabelText("Submitting")).toBeInTheDocument();
  });
});
