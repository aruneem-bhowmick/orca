import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, fireEvent, waitFor } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { Register } from "@/pages/Register";
import * as authApi from "@/api/auth";
import { useAuthStore } from "@/store/auth";

vi.mock("@/api/auth", () => ({
  login: vi.fn(),
  register: vi.fn(),
  logout: vi.fn(),
  getMe: vi.fn(),
}));

describe("Register page", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useAuthStore.getState().clearAuth();
    vi.mocked(authApi.getMe).mockRejectedValue(new Error("No session"));
  });

  it("renders the registration form", () => {
    render(<Register />);
    expect(screen.getByText("Create your Orca account")).toBeInTheDocument();
    expect(screen.getByLabelText("Email")).toBeInTheDocument();
    expect(screen.getByLabelText("Username")).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
  });

  it("shows password strength indicator as user types", async () => {
    render(<Register />);

    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "ab" } });
    expect(screen.getByTestId("strength-label")).toHaveTextContent("Very weak");

    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "abcdefgh" } });
    expect(screen.getByTestId("strength-label")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Password"), {
      target: { value: "AbCdEf12!@" },
    });
    const strengthBars = screen.getAllByTestId("strength-bar");
    expect(strengthBars.length).toBe(5);
  });

  it("shows validation error when password is too short", async () => {
    render(<Register />);

    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "new@example.com" } });
    fireEvent.change(screen.getByLabelText("Username"), { target: { value: "newuser" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "short" } });
    fireEvent.click(screen.getByRole("button", { name: "Create account" }));

    await waitFor(() => {
      expect(screen.getByTestId("register-error")).toHaveTextContent(
        "Password must be at least 8 characters.",
      );
    });
  });

  it("displays 409 error on duplicate email", async () => {
    vi.mocked(authApi.register).mockRejectedValueOnce({
      isAxiosError: true,
      response: { status: 409, data: { detail: "Email already registered" } },
    });

    render(<Register />);

    fireEvent.change(screen.getByLabelText("Email"), {
      target: { value: "existing@example.com" },
    });
    fireEvent.change(screen.getByLabelText("Username"), { target: { value: "existing" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "password123" } });
    fireEvent.click(screen.getByRole("button", { name: "Create account" }));

    await waitFor(() => {
      expect(screen.getByTestId("register-error")).toHaveTextContent("Email already registered");
    });
  });
});
