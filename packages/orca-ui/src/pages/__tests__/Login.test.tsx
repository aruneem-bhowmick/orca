import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, fireEvent, waitFor } from "@testing-library/react";
import { render } from "@/test/test-utils";
import { Login } from "@/pages/Login";
import * as authApi from "@/api/auth";
import { useAuthStore } from "@/store/auth";
import { mockTokenResponse, mockUser } from "@/test/mocks/handlers";

vi.mock("@/api/auth", () => ({
  login: vi.fn(),
  register: vi.fn(),
  logout: vi.fn(),
  getMe: vi.fn(),
}));

describe("Login page", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useAuthStore.getState().clearAuth();
    vi.mocked(authApi.getMe).mockRejectedValue(new Error("No session"));
  });

  it("renders the login form", () => {
    render(<Login />);
    expect(screen.getByText("Sign in to Orca")).toBeInTheDocument();
    expect(screen.getByLabelText("Email")).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sign in" })).toBeInTheDocument();
  });

  it("shows validation error when fields are empty", async () => {
    render(<Login />);
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));

    await waitFor(() => {
      expect(screen.getByTestId("login-error")).toHaveTextContent(
        "Email and password are required.",
      );
    });
  });

  it("calls login API on valid submission", async () => {
    vi.mocked(authApi.login).mockResolvedValueOnce(mockTokenResponse);
    vi.mocked(authApi.getMe).mockResolvedValueOnce(mockUser);

    render(<Login />);

    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "test@example.com" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "password123" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));

    await waitFor(() => {
      expect(authApi.login).toHaveBeenCalledWith({
        email: "test@example.com",
        password: "password123",
      });
    });
  });

  it("displays error on 401 response", async () => {
    vi.mocked(authApi.login).mockRejectedValueOnce({
      response: { status: 401, data: { detail: "Invalid credentials" } },
    });

    render(<Login />);

    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "wrong@example.com" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "wrong" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));

    await waitFor(() => {
      expect(screen.getByTestId("login-error")).toHaveTextContent("Invalid credentials");
    });
  });

  it("renders OAuth buttons", () => {
    render(<Login />);
    expect(screen.getByRole("button", { name: "Google" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "GitHub" })).toBeInTheDocument();
  });
});
