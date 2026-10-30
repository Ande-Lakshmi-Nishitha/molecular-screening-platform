// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { taskReason } from "../api/jobState";
import JobStatus from "./JobStatus";

afterEach(cleanup);

const task = (tier, status, extra = {}) => ({
  id: `t-${tier}`, tier, status, skip_reason: null, error: null, attempts: 1, result: null, ...extra,
});
const job = (tasks, status = "running") => ({ job_id: "j-1", status, tasks });

describe("JobStatus", () => {
  it("shows every tier with its own state (FR-030)", () => {
    render(<JobStatus job={job([task("ml", "done", { result: { model_version: "1.0.0", payload: { logS: -2 } } }), task("dl", "running"), task("qml", "queued")])} />);
    expect(screen.getByTestId("tier-ml-status").textContent).toBe("Done");
    expect(screen.getByTestId("tier-dl-status").textContent).toBe("Running");
    expect(screen.getByTestId("tier-qml-status").textContent).toBe("Queued");
  });

  it("shows a finished tier's result while the others are still running (FR-031)", () => {
    render(<JobStatus job={job([task("ml", "done", { result: { model_version: "1.0.0", payload: { logS: -2.5 } } }), task("dl", "running")])} />);
    expect(screen.getByTestId("tier-ml-result").textContent).toContain("-2.5");
    expect(screen.queryByTestId("tier-dl-result")).toBeNull();
  });

  it("shows why a task failed, with the attempts (FR-032)", () => {
    render(<JobStatus job={job([task("dl", "failed", { error: "timeout", attempts: 3 })], "partial")} />);
    expect(screen.getByTestId("tier-dl-reason").textContent).toBe("timeout (after 3 attempts)");
  });

  it("shows the explanatory reason for a skipped task (BR-003)", () => {
    render(<JobStatus job={job([task("qml", "skipped", { skip_reason: "More than nine heavy atoms." })], "partial")} />);
    expect(screen.getByTestId("tier-qml-reason").textContent).toBe("More than nine heavy atoms.");
  });

  it("never shows a skipped task without a reason", () => {
    expect(taskReason(task("qml", "skipped"))).toMatch(/does not apply/);
    expect(taskReason(task("dl", "failed"))).toMatch(/No reason/);
  });

  it("explains a cancelled task and shows nothing for a running one", () => {
    expect(taskReason(task("dl", "cancelled"))).toMatch(/Cancelled/);
    expect(taskReason(task("dl", "running"))).toBeNull();
  });

  it("renders nothing without a job", () => {
    const { container } = render(<JobStatus job={null} />);
    expect(container.firstChild).toBeNull();
  });
});
