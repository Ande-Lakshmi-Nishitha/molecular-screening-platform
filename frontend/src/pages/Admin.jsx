import { useCallback, useEffect, useState } from "react";
import { api, probe } from "../api/client";
import ErrorBanner from "../components/ErrorBanner";

function Status({ available }) {
  return (
    <span
      style={{
        color: available ? "#067647" : "#b42318",
        fontWeight: "bold",
      }}
    >
      {available ? "Available" : "Unavailable"}
    </span>
  );
}

function Metric({ title, value, description }) {
  return (
    <div
      style={{
        border: "1px solid #d0d5dd",
        borderRadius: 8,
        padding: 16,
        flex: "1 1 180px",
      }}
    >
      <p style={{ color: "#667085", margin: 0 }}>{title}</p>
      <h2 style={{ margin: "8px 0" }}>{value}</h2>
      <small style={{ color: "#667085" }}>{description}</small>
    </div>
  );
}

export default function Admin() {
  const [health, setHealth] = useState(null);
  const [readiness, setReadiness] = useState(null);
  const [models, setModels] = useState(null);
  const [errors, setErrors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);

  const refreshDashboard = useCallback(async () => {
    setLoading(true);
    setErrors([]);

    const [healthResult, readyResult, modelResult] =
      await Promise.all([
        probe("/healthz"),
        probe("/readyz"),
        api.getModels().then(
          (data) => ({ success: true, data }),
          (error) => ({ success: false, error })
        ),
      ]);

    setHealth(healthResult);
    setReadiness(readyResult);

    if (modelResult.success) {
      setModels(modelResult.data);
    } else {
      setModels(null);
      setErrors([modelResult.error]);
    }

    setLastUpdated(new Date());
    setLoading(false);
  }, []);

  useEffect(() => {
    refreshDashboard();
  }, [refreshDashboard]);

  const checks = readiness?.data?.checks || {};
  const modelList = Array.isArray(models)
    ? models
    : models?.models || models?.tiers || [];

  return (
    <div style={{ padding: 20 }}>
      <h1>Admin Dashboard</h1>
      <p>System monitoring and model information.</p>

      {errors.map((error, index) => (
        <ErrorBanner key={index} error={error} />
      ))}

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 12,
        }}
      >
        <div>
          <h2>System Overview</h2>
          <small>
            Last updated:{" "}
            {lastUpdated ? lastUpdated.toLocaleString() : "Not yet updated"}
          </small>
        </div>

        <button onClick={refreshDashboard} disabled={loading}>
          {loading ? "Refreshing..." : "Refresh dashboard"}
        </button>
      </div>

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 12,
          margin: "16px 0 28px",
        }}
      >
        <Metric
          title="Backend"
          value={
            loading ? "Checking..." :
            health?.available ? "Online" : "Offline"
          }
          description="Flask liveness check"
        />

        <Metric
          title="Database"
          value={loading ? "Checking..." : checks.postgres || "Unknown"}
          description="PostgreSQL connectivity"
        />

        <Metric
          title="Redis"
          value={loading ? "Checking..." : checks.redis || "Unknown"}
          description="Redis connectivity"
        />

        <Metric
          title="Model API"
          value={
            loading ? "Checking..." :
            models !== null ? "Responding" : "Unavailable"
          }
          description="Model information endpoint"
        />
      </div>

      <section style={{ marginBottom: 28 }}>
        <h2>Service Health</h2>

        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th align="left">Service</th>
              <th align="left">Status</th>
              <th align="left">Details</th>
            </tr>
          </thead>

          <tbody>
            <tr>
              <td>Flask Backend</td>
              <td>
                {loading ? "Checking..." :
                  <Status available={health?.available} />}
              </td>
              <td>GET /healthz</td>
            </tr>

            <tr>
              <td>PostgreSQL</td>
              <td>
                {loading ? "Checking..." :
                  <Status available={checks.postgres === "ok"} />}
              </td>
              <td>Database readiness check</td>
            </tr>

            <tr>
              <td>Redis</td>
              <td>
                {loading ? "Checking..." :
                  <Status available={checks.redis === "ok"} />}
              </td>
              <td>Redis readiness check</td>
            </tr>
          </tbody>
        </table>

        {readiness && (
          <p>
            Overall readiness:{" "}
            <Status available={readiness.available} />
            {" "}(HTTP {readiness.status})
          </p>
        )}
      </section>

      <section>
        <h2>Model Information</h2>

        {loading && <p>Loading model information...</p>}

        {!loading && models === null && (
          <p>Model information could not be retrieved.</p>
        )}

        {!loading && modelList.length > 0 && (
          <table style={{ width: "100%" }}>
            <thead>
              <tr>
                <th align="left">Model</th>
                <th align="left">Information</th>
              </tr>
            </thead>

            <tbody>
              {modelList.map((model, index) => (
                <tr key={model.id || model.name || index}>
                  <td>{model.name || model.id || `Model ${index + 1}`}</td>
                  <td>
                    <pre style={{ whiteSpace: "pre-wrap" }}>
                      {JSON.stringify(model, null, 2)}
                    </pre>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {!loading && models !== null && modelList.length === 0 && (
          <pre style={{ whiteSpace: "pre-wrap" }}>
            {JSON.stringify(models, null, 2)}
          </pre>
        )}
      </section>
    </div>
  );
}
