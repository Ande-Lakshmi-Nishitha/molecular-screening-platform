// components/JobStatus.jsx - owner: Abheetha
// The triage summary, then one row per tier: its state, its formatted result once done, and the reason
// when it failed or was skipped (MSP-36, MSP-37).
import { stateLabel, taskReason, tierLabel } from "../api/jobState";
import TierResult from "./TierResult";
import TriageSummary from "./TriageSummary";

const BADGE_COLOURS = {
  queued: "#475467",
  running: "#175cd3",
  done: "#067647",
  failed: "#b42318",
  skipped: "#b54708",
  cancelled: "#475467",
};

function Badge({ status, testId }) {
  return (
    <span
      data-testid={testId}
      style={{ background: BADGE_COLOURS[status] || "#475467", color: "#fff", padding: "2px 8px", borderRadius: 10, fontSize: 12 }}
    >
      {stateLabel(status)}
    </span>
  );
}

function TierRow({ task }) {
  const reason = taskReason(task);
  return (
    <li data-testid={`tier-${task.tier}`} style={{ border: "1px solid #d0d5dd", borderRadius: 6, padding: "10px 12px", margin: "8px 0", listStyle: "none" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <strong>{tierLabel(task.tier)}</strong>
        <Badge status={task.status} testId={`tier-${task.tier}-status`} />
      </div>
      {reason && (
        <p data-testid={`tier-${task.tier}-reason`} style={{ margin: "6px 0 0" }}>
          {reason}
        </p>
      )}
      {task.status === "done" && task.result && (
        <div data-testid={`tier-${task.tier}-result`} style={{ marginTop: 8 }}>
          <div style={{ fontSize: 12, color: "#475467" }}>Model version {task.result.model_version}</div>
          <TierResult tier={task.tier} payload={task.result.payload} />
        </div>
      )}
    </li>
  );
}

export default function JobStatus({ job }) {
  if (!job) return null;
  return (
    <section>
      <p aria-live="polite">
        Job status: <strong data-testid="job-status">{stateLabel(job.status)}</strong>
      </p>
      <p style={{ fontSize: 12, color: "#475467" }}>Job {job.job_id}</p>
      <TriageSummary summary={job.summary} />
      <ul style={{ padding: 0 }}>
        {job.tasks.map((task) => (
          <TierRow key={task.id} task={task} />
        ))}
      </ul>
    </section>
  );
}
