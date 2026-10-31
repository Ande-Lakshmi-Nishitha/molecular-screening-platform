// Small, pure helpers for job and task states (FR-030). No React in here, so they are easy to test.

export const POLL_INTERVAL_MS = 2000;

// A job in one of these states can still change, so the tracker keeps polling.
export const ACTIVE_STATUSES = ["queued", "running"];

export const isActive = (status) => ACTIVE_STATUSES.includes(status);

// Cancel is offered only while the job can still change (FR-056).
// The API refuses jobs that belong to someone else, so the caller's own jobs are the only ones shown.
export const canCancel = (job) => Boolean(job) && isActive(job.status);

const TIER_LABELS = { ml: "Machine learning", dl: "Deep learning", qml: "Quantum (QML)" };
export const tierLabel = (tier) => TIER_LABELS[tier] || tier;

// Plain-language text for each task state; the label is always shown, never colour alone.
export const STATE_LABELS = {
  queued: "Queued",
  running: "Running",
  done: "Done",
  failed: "Failed",
  skipped: "Skipped",
  cancelled: "Cancelled",
  // Job-level only (the whole job, not one tier):
  partial: "Partly complete",
  complete: "Complete",
};
export const stateLabel = (status) => STATE_LABELS[status] || status;

// Reason shown for a task that did not produce a result (FR-032, BR-003).
export function taskReason(task) {
  if (task.status === "skipped") {
    return task.skip_reason || "Skipped: this tier does not apply to this molecule.";
  }
  if (task.status === "failed") {
    const attempts = task.attempts > 1 ? ` (after ${task.attempts} attempts)` : "";
    return `${task.error || "No reason was recorded."}${attempts}`;
  }
  if (task.status === "cancelled") return "Cancelled before it finished.";
  return null;
}
