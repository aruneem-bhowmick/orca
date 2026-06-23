import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { Landing } from "@/pages/Landing";
import apiClient from "@/api/client";
import { mockHealthStatus } from "@/test/mocks/handlers";

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
    vi.mocked(apiClient.get).mockResolvedValue({ data: mockHealthStatus });
  });

  it("renders the hero section", () => {
    render(<Landing />);
    expect(screen.getByText("Meta-Learning Platform")).toBeInTheDocument();
    expect(screen.getByText("Start building")).toBeInTheDocument();
  });

  it("renders three service cards", () => {
    render(<Landing />);
    expect(screen.getByText("OrcaMind")).toBeInTheDocument();
    expect(screen.getByText("OrcaLab")).toBeInTheDocument();
    expect(screen.getByText("OrcaNet")).toBeInTheDocument();
  });

  it("renders navigation links", () => {
    render(<Landing />);
    expect(screen.getByText("Sign in")).toBeInTheDocument();
    expect(screen.getByText("Get started")).toBeInTheDocument();
  });
});
