// pages/JobTracker.jsx - owner: Abheetha
// Polls one job, shows each tier as it finishes, and lets the owner cancel it (MSP-36).
import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import { POLL_INTERVAL_MS, canCancel, isActive } from "../api/jobState";
import ErrorBanner from "../components/ErrorBanner";
import JobStatus from "../components/JobStatus";

export default function JobTracker() {
  const { id } = useParams();
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null); // problems loading the job
  const [cancelError, setCancelError] = useState(null); // problems cancelling it; survives the refresh
  const [confirming, setConfirming] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [reload, setReload] = useState(0);
  const timer = useRef(null);

  useEffect(() => {
    if (!id) return undefined;
    let stopped = false;

    async function poll() {
      try {
        const data = await api.getJob(id);
        if (stopped) return;
        setJob(data);
        setError(null);
        // Keep polling only while the job can still change.
        if (isActive(data.status)) timer.current = setTimeout(poll, POLL_INTERVAL_MS);
      } catch (err) {
        if (stopped) return;
        setError(err);
        // A dropped connection or a server error may pass; retry more slowly. 4xx errors stop here.
        if (err.status === 0 || err.status >= 500) {
          timer.current = setTimeout(poll, POLL_INTERVAL_MS * 2);
        }
      }
    }

    poll();
    return () => {
      stopped = true;
      clearTimeout(timer.current);
    };
  }, [id, reload]);

  async function confirmCancel() {
    setCancelling(true);
    setCancelError(null);
    try {
      setJob(await api.cancelJob(id));
    } catch (err) {
      setCancelError(err);
    } finally {
      setCancelling(false);
      setConfirming(false);
      setReload((n) => n + 1); // refresh from the server whatever the outcome
    }
  }

  if (!id) {
    return (
      <div>
        <h1>Job tracker</h1>
        <p>
          Submit a molecule to start a job, or open one from <Link to="/history">History</Link>.
        </p>
        <Link to="/submit">Submit a molecule</Link>
      </div>
    );
  }

  return (
    <div>
      <h1>Job tracker</h1>
      <ErrorBanner error={error} />
      <ErrorBanner error={cancelError} />
      {!job && !error && <p>Loading job…</p>}
      <JobStatus job={job} />
      {canCancel(job) && !confirming && (
        <button type="button" data-testid="cancel-button" onClick={() => setConfirming(true)}>
          Cancel job
        </button>
      )}
      {canCancel(job) && confirming && (
        <div role="group" aria-label="Confirm cancel">
          <span>Cancel this job? Finished results are kept.</span>{" "}
          <button type="button" data-testid="cancel-confirm" onClick={confirmCancel} disabled={cancelling}>
            {cancelling ? "Cancelling…" : "Yes, cancel"}
          </button>{" "}
          <button type="button" data-testid="cancel-keep" onClick={() => setConfirming(false)} disabled={cancelling}>
            Keep running
          </button>
        </div>
      )}
    </div>
  );
}
