
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import ErrorBanner from "../components/ErrorBanner";
import TierResult from "../components/TierResult";

const TIER_ORDER = ["ml", "dl", "qml"];

const TIER_LABELS = {
  ml: "Machine Learning",
  dl: "Deep Learning",
  qml: "Quantum Machine Learning",
};

function getTask(job, tier) {
  return job.tasks?.find((task) => task.tier === tier);
}

function MoleculeName(job) {
  return job.molecule?.name || job.molecule?.formula || "Unknown molecule";
}

export default function Compare() {
  const [jobs, setJobs] = useState([]);
  const [selected, setSelected] = useState([]);
  const [compared, setCompared] = useState([]);
  const [loading, setLoading] = useState(true);
  const [comparing, setComparing] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function loadJobs() {
      try {
        const data = await api.listJobs(1, 20);

        if (!cancelled) {
          setJobs(data.jobs || []);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) setError(err);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadJobs();

    return () => {
      cancelled = true;
    };
  }, []);

  function toggleJob(jobId) {
    setSelected((current) =>
      current.includes(jobId)
        ? current.filter((id) => id !== jobId)
        : [...current, jobId]
    );
  }

  async function compareJobs() {
    if (selected.length < 2) return;

    setComparing(true);
    setError(null);
    setCompared([]);

    try {
      const results = await Promise.all(
        selected.map((id) => api.getJob(id))
      );

      setCompared(results);
    } catch (err) {
      setError(err);
    } finally {
      setComparing(false);
    }
  }

  return (
    <div>
      <h1>Compare Molecules</h1>

      <p>
        Select at least two screening jobs to compare their
        molecular properties and ML, DL and QML predictions.
      </p>

      <ErrorBanner error={error} />

      <h2>Select molecules</h2>

      {loading && <p>Loading previous jobs...</p>}

      {!loading && jobs.length === 0 && !error && (
        <p>
          No previous jobs found.
          <Link to="/submit"> Submit a molecule</Link> to get started.
        </p>
      )}

      {jobs.length > 0 && (
        <>
          <table>
            <thead>
              <tr>
                <th>Select</th>
                <th>Molecule</th>
                <th>Formula</th>
                <th>Job status</th>
                <th>View</th>
              </tr>
            </thead>

            <tbody>
              {jobs.map((job) => (
                <tr key={job.job_id}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selected.includes(job.job_id)}
                      onChange={() => toggleJob(job.job_id)}
                      aria-label={`Select ${MoleculeName(job)}`}
                    />
                  </td>

                  <td>{MoleculeName(job)}</td>

                  <td>{job.molecule?.formula || "N/A"}</td>

                  <td>{job.status}</td>

                  <td>
                    <Link to={`/jobs/${job.job_id}`}>View job</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <p>
            {selected.length} molecule(s) selected.
            Select at least two to compare.
          </p>

          <button
            type="button"
            onClick={compareJobs}
            disabled={selected.length < 2 || comparing}
          >
            {comparing ? "Comparing..." : "Compare selected molecules"}
          </button>
        </>
      )}

      {compared.length > 0 && (
        <section style={{ marginTop: 30 }}>
          <h2>Comparison Results</h2>

          <div style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>Parameter</th>

                  {compared.map((job) => (
                    <th key={job.job_id}>
                      {MoleculeName(job)}
                      <div style={{ fontSize: 11 }}>
                        Job: {job.job_id}
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>

              <tbody>
                <tr>
                  <th>Formula</th>
                  {compared.map((job) => (
                    <td key={job.job_id}>
                      {job.molecule?.formula || "N/A"}
                    </td>
                  ))}
                </tr>

                <tr>
                  <th>SMILES</th>
                  {compared.map((job) => (
                    <td key={job.job_id}>
                      <code>
                        {job.molecule?.canonical_smiles || "N/A"}
                      </code>
                    </td>
                  ))}
                </tr>

                <tr>
                  <th>Overall status</th>
                  {compared.map((job) => (
                    <td key={job.job_id}>{job.status}</td>
                  ))}
                </tr>

                {TIER_ORDER.map((tier) => (
                  <tr key={tier}>
                    <th>{TIER_LABELS[tier]}</th>

                    {compared.map((job) => {
                      const task = getTask(job, tier);

                      return (
                        <td
                          key={job.job_id}
                          style={{
                            verticalAlign: "top",
                            minWidth: 220,
                            padding: 12,
                            border: "1px solid #d0d5dd",
                          }}
                        >
                          {!task && <p>Not available</p>}

                          {task && (
                            <>
                              <strong>Status: {task.status}</strong>

                              {task.result && (
                                <>
                                  <p style={{ fontSize: 12 }}>
                                    Model version:{" "}
                                    {task.result.model_version || "N/A"}
                                  </p>

                                  <TierResult
                                    tier={tier}
                                    payload={task.result.payload}
                                  />
                                </>
                              )}

                              {task.status !== "done" && (
                                <p>
                                  No completed result available.
                                </p>
                              )}
                            </>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}

