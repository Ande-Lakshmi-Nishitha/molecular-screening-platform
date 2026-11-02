// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import Admin from "./Admin";

vi.mock("../api/client", () => ({ api: { getModels: vi.fn() }, probe: vi.fn() }));
import { api, probe } from "../api/client";

const up = (data) => ({ available: true, status: 200, data });
const READY = up({ status: "ready", checks: { postgres: "ok", redis: "ok" } });

beforeEach(() => {
  vi.clearAllMocks();
  api.getModels.mockResolvedValue({ models: [] });
});
afterEach(cleanup);

describe("Admin dashboard", () => {
  it("asks for the probes on the same origin, not a hard-coded address", async () => {
    probe.mockResolvedValue(READY);
    render(<Admin />);
    await screen.findByText("Online");
    expect(probe).toHaveBeenCalledWith("/healthz");
    expect(probe).toHaveBeenCalledWith("/readyz");
  });

  it("shows every service as available when the backend, database and Redis answer", async () => {
    probe.mockImplementation((path) => Promise.resolve(path === "/healthz" ? up({ status: "ok" }) : READY));
    render(<Admin />);
    expect(await screen.findByText("Online")).toBeTruthy();
    expect(screen.getAllByText("Available").length).toBe(4); // three services plus overall readiness
    expect(document.body.textContent).toContain("(HTTP 200)");
  });

  it("shows the backend as offline when it cannot be reached (HTTP 0)", async () => {
    probe.mockResolvedValue({ available: false, status: 0, data: {}, message: "Failed to fetch" });
    render(<Admin />);
    expect(await screen.findByText("Offline")).toBeTruthy();
    expect(document.body.textContent).toContain("(HTTP 0)");
    expect(screen.queryByText("Online")).toBeNull();
  });

  it("shows which dependency is down when the backend answers 503", async () => {
    probe.mockImplementation((path) =>
      Promise.resolve(
        path === "/healthz"
          ? up({ status: "ok" })
          : { available: false, status: 503, data: { status: "not_ready", checks: { postgres: "ok", redis: "unavailable" } } },
      ),
    );
    render(<Admin />);
    expect(await screen.findByText("unavailable")).toBeTruthy(); // the Redis card
    expect(screen.getByText("Online")).toBeTruthy(); // the process itself is up
    expect(document.body.textContent).toContain("(HTTP 503)");
  });
});
