// components/TriageSummary.jsx - owner: Abheetha
// The job's triage summary (MSP-37; FR-036). It is written by the aggregator (MSP-31); until that lands,
// jobs.summary is empty and this says so. Entries are shown generically until the summary's shape is agreed.
import { Fragment } from "react";

const label = (key) => key.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
const show = (value) => (typeof value === "string" || typeof value === "number" ? String(value) : JSON.stringify(value));

export default function TriageSummary({ summary }) {
  if (!summary || typeof summary !== "object" || Object.keys(summary).length === 0) {
    return (
      <p data-testid="triage-pending" style={{ fontSize: 12, color: "#475467" }}>
        The triage summary is not available yet.
      </p>
    );
  }
  return (
    <section data-testid="triage-summary">
      <h2>Triage summary</h2>
      <dl>
        {Object.entries(summary).map(([key, value]) => (
          <Fragment key={key}>
            <dt>{label(key)}</dt>
            <dd data-testid={`triage-${key}`}>{show(value)}</dd>
          </Fragment>
        ))}
      </dl>
    </section>
  );
}
