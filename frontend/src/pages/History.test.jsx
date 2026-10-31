// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import History, { PAGE_SIZE } from "./History";

vi.mock("../api/client", () => ({ api: { listJobs: vi.fn() } }));
import { api } from "../api/client";

const job = (id, status = "complete", extra = {}) => ({
  job_id: id,
  status,
  target_id: "EGFR",
  created_at: "2026-11-02T10:00:00+00:00",
  finished_at: null,
  molecule: { canonical_smiles: "CCO", formula: "C2H6O", inchikey: "LFQSCWFLJHTTHZ-UHFFFAOYSA-N" },
  tiers: { ml: "done", dl: "done", qml: "skipped" },
  ...extra,
});
const body = (jobs, total = jobs.length, page = 1) => ({ jobs, page, page_size: PAGE_SIZE, total });

const renderHistory = () =>
  render(
    <MemoryRouter>
      <History />
    </MemoryRouter>,
  );

beforeEach(() => vi.clearAllMocks());
afterEach(cleanup);

describe("History", () => {
  it("lists the jobs with their status and a link to each one", async () => {
    api.listJobs.mockResolvedValue(body([job("j-1"), job("j-2", "partial")]));
    renderHistory();

    expect((await screen.findByTestId("history-status-j-1")).textContent).toBe("Complete");
    expect(screen.getByTestId("history-status-j-2").textContent).toBe("Partial");
    expect(screen.getByTestId("history-link-j-1").getAttribute("href")).toBe("/jobs/j-1");
    expect(screen.getByTestId("history-total").textContent).toBe("2 jobs");
    expect(api.listJobs).toHaveBeenCalledWith(1, PAGE_SIZE);
  });

  it("pages forward and back", async () => {
    api.listJobs.mockResolvedValue(body([job("j-1")], PAGE_SIZE + 5));
    renderHistory();
    await screen.findByTestId("history-row-j-1");
    expect(screen.getByTestId("history-page").textContent).toBe("Page 1 of 2");
    expect(screen.getByTestId("history-prev").disabled).toBe(true);

    api.listJobs.mockResolvedValue(body([job("j-9")], PAGE_SIZE + 5, 2));
    fireEvent.click(screen.getByTestId("history-next"));
    await screen.findByTestId("history-row-j-9");
    expect(api.listJobs).toHaveBeenLastCalledWith(2, PAGE_SIZE);
    expect(screen.getByTestId("history-next").disabled).toBe(true);
    expect(screen.getByTestId("history-prev").disabled).toBe(false);
  });

  it("shows a prompt when the user has no jobs", async () => {
    api.listJobs.mockResolvedValue(body([], 0));
    renderHistory();
    expect(await screen.findByTestId("history-empty")).toBeTruthy();
    expect(screen.queryByTestId("history-next")).toBeNull();
  });

  it("shows the API error", async () => {
    api.listJobs.mockRejectedValue(new ApiError({ status: 500, code: "INTERNAL", message: "Server error.", requestId: "req-5" }));
    renderHistory();
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("Server error.");
    expect(alert.textContent).toContain("req-5");
  });
});
