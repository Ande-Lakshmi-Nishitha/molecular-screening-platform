// pages/History.jsx - owner: Abheetha
// The caller's jobs, newest first, paged (MSP-37; FR-049).
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import ErrorBanner from "../components/ErrorBanner";

export const PAGE_SIZE = 20;

const STATUS_LABEL = {
  queued: "Queued",
  running: "Running",
  complete: "Complete",
  partial: "Partial",
  failed: "Failed",
  cancelled: "Cancelled",
};
const TIER_LABEL = { ml: "ML", dl: "DL", qml: "QML" };

const shorten = (text, max = 40) => (text && text.length > max ? `${text.slice(0, max)}…` : text || "");

export default function History() {
  const [page, setPage] = useState(1);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let stopped = false;
    api
      .listJobs(page, PAGE_SIZE)
      .then((body) => {
        if (stopped) return;
        setData(body);
        setError(null);
      })
      .catch((err) => {
        if (!stopped) setError(err);
      });
    return () => {
      stopped = true;
    };
  }, [page]);

  const total = data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div>
      <h1>Job history</h1>
      <ErrorBanner error={error} />
      {!data && !error && <p>Loading jobs…</p>}
      {data && data.jobs.length === 0 && (
        <p data-testid="history-empty">
          You have no jobs yet. <Link to="/submit">Submit a molecule</Link>.
        </p>
      )}
      {data && data.jobs.length > 0 && (
        <>
          <p data-testid="history-total">{total} job{total === 1 ? "" : "s"}</p>
          <table>
            <thead>
              <tr>
                <th>Submitted</th>
                <th>Molecule</th>
                <th>Target</th>
                <th>Status</th>
                <th>Tiers</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {data.jobs.map((job) => (
                <tr key={job.job_id} data-testid={`history-row-${job.job_id}`}>
                  <td>{job.created_at ? new Date(job.created_at).toLocaleString() : ""}</td>
                  <td title={job.molecule.canonical_smiles}>
                    {job.molecule.formula} <code>{shorten(job.molecule.canonical_smiles)}</code>
                  </td>
                  <td>{job.target_id || "none"}</td>
                  <td data-testid={`history-status-${job.job_id}`}>{STATUS_LABEL[job.status] || job.status}</td>
                  <td>
                    {Object.entries(job.tiers).map(([tier, status]) => (
                      <span key={tier} style={{ marginRight: 8 }}>
                        {TIER_LABEL[tier] || tier}: {status}
                      </span>
                    ))}
                  </td>
                  <td>
                    <Link to={`/jobs/${job.job_id}`} data-testid={`history-link-${job.job_id}`}>
                      Open
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: 12 }}>
            <button type="button" data-testid="history-prev" onClick={() => setPage((p) => p - 1)} disabled={page <= 1}>
              Previous
            </button>{" "}
            <span data-testid="history-page">
              Page {page} of {pages}
            </span>{" "}
            <button type="button" data-testid="history-next" onClick={() => setPage((p) => p + 1)} disabled={page >= pages}>
              Next
            </button>
          </div>
        </>
      )}
    </div>
  );
}
