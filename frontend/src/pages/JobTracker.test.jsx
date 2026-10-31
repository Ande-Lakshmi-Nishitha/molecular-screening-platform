// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import { POLL_INTERVAL_MS } from "../api/jobState";
import JobTracker from "./JobTracker";

vi.mock("../api/client", () => ({ api: { getJob: vi.fn(), cancelJob: vi.fn() } }));
import { api } from "../api/client";

const task = (tier, status, extra = {}) => ({
  id: `t-${tier}`, tier, status, skip_reason: null, error: null, attempts: 1, result: null, ...extra,
});
const job = (status, tasks) => ({ job_id: "j-1", status, tasks });
const RUNNING = job("running", [
  task("ml", "done", { result: { model_version: "1.0.0", payload: { logS: -2.5 } } }),
  task("dl", "running"),
  task("qml", "queued"),
]);
const COMPLETE = job("complete", [
  task("ml", "done", { result: { model_version: "1.0.0", payload: { logS: -2.5 } } }),
  task("dl", "done", { result: { model_version: "1.0.0", payload: { pIC50: 6.1 } } }),
  task("qml", "skipped", { skip_reason: "More than nine heavy atoms." }),
]);

function renderAt(path) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/jobs" element={<JobTracker />} />
        <Route path="/jobs/:id" element={<JobTracker />} />
      </Routes>
    </MemoryRouter>,
  );
}
const tick = (ms) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });

beforeEach(() => { vi.useFakeTimers(); vi.clearAllMocks(); });
afterEach(() => { cleanup(); vi.useRealTimers(); });

describe("JobTracker", () => {
  it("shows partial results first, then the rest, and stops polling when the job is final", async () => {
    api.getJob.mockResolvedValueOnce(RUNNING).mockResolvedValue(COMPLETE);
    renderAt("/jobs/j-1");
    await tick(0);
    expect(screen.getByTestId("tier-ml-result")).toBeTruthy();
    expect(screen.queryByTestId("tier-dl-result")).toBeNull();
    expect(screen.getByTestId("cancel-button")).toBeTruthy();

    await tick(POLL_INTERVAL_MS);
    expect(screen.getByTestId("tier-dl-result")).toBeTruthy();
    expect(screen.getByTestId("tier-qml-reason").textContent).toBe("More than nine heavy atoms.");
    expect(screen.queryByTestId("cancel-button")).toBeNull();

    const calls = api.getJob.mock.calls.length;
    await tick(POLL_INTERVAL_MS * 5);
    expect(api.getJob.mock.calls.length).toBe(calls);
  });

  it("stops polling when the user leaves the page", async () => {
    api.getJob.mockResolvedValue(RUNNING);
    const { unmount } = renderAt("/jobs/j-1");
    await tick(0);
    unmount();
    const calls = api.getJob.mock.calls.length;
    await tick(POLL_INTERVAL_MS * 5);
    expect(api.getJob.mock.calls.length).toBe(calls);
  });

  it("asks for confirmation, then cancels and shows the cancelled tiers (FR-056)", async () => {
    const cancelled = job("cancelled", [
      task("ml", "done", { result: { model_version: "1.0.0", payload: { logS: -2.5 } } }),
      task("dl", "cancelled"),
      task("qml", "cancelled"),
    ]);
    api.getJob.mockResolvedValueOnce(RUNNING).mockResolvedValue(cancelled);
    api.cancelJob.mockResolvedValue(cancelled);
    renderAt("/jobs/j-1");
    await tick(0);

    fireEvent.click(screen.getByTestId("cancel-button"));
    expect(api.cancelJob).not.toHaveBeenCalled();
    await act(async () => { fireEvent.click(screen.getByTestId("cancel-confirm")); });
    await tick(0);

    expect(api.cancelJob).toHaveBeenCalledWith("j-1");
    expect(screen.getByTestId("job-status").textContent).toBe("Cancelled");
    expect(screen.getByTestId("tier-dl-reason").textContent).toMatch(/Cancelled/);
    expect(screen.getByTestId("tier-ml-result")).toBeTruthy();
    expect(screen.queryByTestId("cancel-button")).toBeNull();
  });

  it("keeps the job running when the user backs out of the confirmation", async () => {
    api.getJob.mockResolvedValue(RUNNING);
    renderAt("/jobs/j-1");
    await tick(0);
    fireEvent.click(screen.getByTestId("cancel-button"));
    fireEvent.click(screen.getByTestId("cancel-keep"));
    expect(api.cancelJob).not.toHaveBeenCalled();
    expect(screen.getByTestId("cancel-button")).toBeTruthy();
  });

  it("shows the API error with its request ID and stops on a 404", async () => {
    api.getJob.mockRejectedValue(new ApiError({ status: 404, code: "JOB_NOT_FOUND", message: "Job not found.", requestId: "req-42" }));
    renderAt("/jobs/missing");
    await tick(0);
    expect(screen.getByRole("alert").textContent).toContain("Job not found.");
    expect(screen.getByRole("alert").textContent).toContain("req-42");
    const calls = api.getJob.mock.calls.length;
    await tick(POLL_INTERVAL_MS * 5);
    expect(api.getJob.mock.calls.length).toBe(calls);
  });

  it("shows a 409 from cancel and refreshes from the server", async () => {
    api.getJob.mockResolvedValue(RUNNING);
    api.cancelJob.mockRejectedValue(new ApiError({ status: 409, code: "JOB_NOT_CANCELLABLE", message: "Every task of this job has already finished." }));
    renderAt("/jobs/j-1");
    await tick(0);
    fireEvent.click(screen.getByTestId("cancel-button"));
    const before = api.getJob.mock.calls.length;
    await act(async () => { fireEvent.click(screen.getByTestId("cancel-confirm")); });
    await tick(0);
    expect(screen.getByRole("alert").textContent).toContain("already finished");
    expect(api.getJob.mock.calls.length).toBeGreaterThan(before);
  });

  it("retries after a dropped connection", async () => {
    api.getJob
      .mockRejectedValueOnce(new ApiError({ status: 0, code: "NETWORK_ERROR", message: "Cannot reach the server." }))
      .mockResolvedValue(COMPLETE);
    renderAt("/jobs/j-1");
    await tick(0);
    expect(screen.getByRole("alert")).toBeTruthy();
    await tick(POLL_INTERVAL_MS * 2);
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByTestId("job-status").textContent).toBe("Complete");
  });

  it("shows a prompt instead of polling when there is no job ID", async () => {
    renderAt("/jobs");
    await tick(0);
    expect(api.getJob).not.toHaveBeenCalled();
    expect(screen.getAllByText(/Submit a molecule/i).length).toBeGreaterThan(0);
  });
});
