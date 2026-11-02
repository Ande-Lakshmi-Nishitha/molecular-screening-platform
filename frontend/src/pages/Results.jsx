
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import ErrorBanner from "../components/ErrorBanner";
import JobStatus from "../components/JobStatus";

export default function Results() {
  const { id } = useParams();

  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [reload, setReload] = useState(0);

  useEffect(() => {
    if (!id) {
      setLoading(false);
      return undefined;
    }

    let cancelled = false;

    async function fetchResults() {
      setLoading(true);
      setError(null);

      try {
        const data = await api.getJob(id);

        if (!cancelled) {
          setJob(data);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    fetchResults();

    return () => {
      cancelled = true;
    };
  }, [id, reload]);

  if (!id) {
    return (
      <div>
        <h1>Screening Results</h1>
        <p>No job selected. Open a job to view its results.</p>
        <Link to="/history">View job history</Link>
      </div>
    );
  }

  return (
    <div>
      <h1>Screening Results</h1>

      <p>
        View the ML, DL and QML screening results
        for your submitted molecule.
      </p>

      <ErrorBanner error={error} />

      {loading && !job && <p>Loading results...</p>}

      {!loading && !job && !error && (
        <p>No results found for this job.</p>
      )}

      {job && <JobStatus job={job} />}

      <div style={{ display: "flex", gap: 12, marginTop: 20 }}>
        <button
          type="button"
          onClick={() => setReload((n) => n + 1)}
          disabled={loading}
        >
          {loading ? "Refreshing..." : "Refresh results"}
        </button>

        <Link to={`/jobs/${id}`}>Back to Job Tracker</Link>
      </div>
    </div>
  );
}

