import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { Landing } from "@/pages/Landing";
import apiClient from "@/api/client";
import { mockHealthStatus, mockDashboardStats } from "@/test/mocks/handlers";

vi.mock("@/api/client", () => ({
  default: {
    get: vi.fn(),
    interceptors: {
      request: { use: vi.fn(), handlers: [] },
      response: { use: vi.fn(), handlers: [] },
    },
  },
}));

vi.mock("@/api/auth", () => ({
  getMe: vi.fn().mockRejectedValue(new Error("No session")),
}));

describe("Landing page", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(apiClient.get).mockImplementation((url: string) => {
      if (url === "/health") {
        return Promise.resolve({ data: mockHealthStatus });
      }
      if (url === "/dashboard/stats") {
        return Promise.resolve({ data: mockDashboardStats });
      }
      return Promise.resolve({ data: {} });
    });
  });

  it("renders the hero section with updated headline", () => {
    render(<Landing />);
    expect(
      screen.getByText("Orca: Meta-Learning Platform"),
    ).toBeInTheDocument();
    expect(screen.getAllByText("Get Started").length).toBeGreaterThanOrEqual(1);
  });

  it("renders three service cards with icons", () => {
    render(<Landing />);
    expect(screen.getByText("OrcaMind")).toBeInTheDocument();
    expect(screen.getByText("OrcaLab")).toBeInTheDocument();
    expect(screen.getByText("OrcaNet")).toBeInTheDocument();
    expect(screen.getByTestId("icon-orcamind")).toBeInTheDocument();
    expect(screen.getByTestId("icon-orcalab")).toBeInTheDocument();
    expect(screen.getByTestId("icon-orcanet")).toBeInTheDocument();
  });

  it("renders navigation links", () => {
    render(<Landing />);
    expect(screen.getAllByText("Sign In").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Get Started").length).toBeGreaterThanOrEqual(1);
  });

  it("renders health status indicators when data is available", async () => {
    render(<Landing />);

    await waitFor(() => {
      expect(screen.getByTestId("status-orcamind")).toBeInTheDocument();
      expect(screen.getByTestId("status-orcalab")).toBeInTheDocument();
      expect(screen.getByTestId("status-orcanet")).toBeInTheDocument();
    });
  });

  it("renders live stats section when data is available", async () => {
    render(<Landing />);

    await waitFor(() => {
      expect(screen.getByTestId("stat-tasks")).toHaveTextContent("42");
      expect(screen.getByTestId("stat-experiments")).toHaveTextContent("128");
      expect(screen.getByTestId("stat-transfers")).toHaveTextContent("56");
    });
  });

  it("renders the live stats section heading", () => {
    render(<Landing />);
    expect(screen.getByText("Platform at a Glance")).toBeInTheDocument();
  });

  it("renders the footer with documentation and GitHub links", () => {
    render(<Landing />);
    expect(screen.getByTestId("landing-footer")).toBeInTheDocument();
    expect(screen.getByText("Documentation")).toBeInTheDocument();
    expect(screen.getByTestId("github-link")).toBeInTheDocument();
  });
});
