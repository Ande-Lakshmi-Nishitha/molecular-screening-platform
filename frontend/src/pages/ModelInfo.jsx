// pages/ModelInfo.jsx - owner: Abheetha
// Version, training-data source, licence and validation metrics of every model (MSP-38; FR-040, BR-009).
import { useEffect, useState } from "react";
import { api } from "../api/client";
import ErrorBanner from "../components/ErrorBanner";

const TIER_LABEL = { ml: "ML", dl: "DL", qml: "QML" };

// Nested metric objects become dotted rows, e.g. "test.mae".
function metricRows(value, prefix = "") {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return Object.entries(value).flatMap(([key, item]) => metricRows(item, prefix ? `${prefix}.${key}` : key));
  }
  const shown = typeof value === "number" && !Number.isInteger(value) ? value.toFixed(3) : String(value);
  return [[prefix, shown]];
}

function Row({ label, id, value }) {
  if (value === null || value === undefined || value === "") return null;
  return (
    <>
      <dt>{label}</dt>
      <dd data-testid={id}>{value}</dd>
    </>
  );
}

function ModelCard({ model }) {
  const { id, data = {}, split = {} } = model;
  const rows = metricRows(model.metrics || {});
  return (
    <section data-testid={`model-${id}`} style={{ border: "1px solid #d0d5dd", borderRadius: 6, padding: 12, margin: "12px 0" }}>
      <h2>
        {model.name} <small data-testid={`model-${id}-version`}>v{model.version}</small>
      </h2>
      <dl>
        <Row label="Tier" id={`model-${id}-tier`} value={TIER_LABEL[model.tier] || model.tier} />
        <Row label="Task" id={`model-${id}-task`} value={model.task} />
        <Row label="Target" id={`model-${id}-target`} value={model.target} />
        <Row label="Unit" id={`model-${id}-unit`} value={model.unit} />
        <Row label="Trained" id={`model-${id}-trained`} value={model.trained_at} />
        <Row label="Training data" id={`model-${id}-source`} value={data.source} />
        <Row label="Data release" id={`model-${id}-release`} value={data.chembl_release} />
        <Row label="Citation" id={`model-${id}-citation`} value={data.citation} />
        <Row label="Data licence" id={`model-${id}-licence`} value={data.licence_note} />
        <Row label="Training molecules" id={`model-${id}-n-train`} value={split.n_train} />
        <Row label="Test molecules" id={`model-${id}-n-test`} value={split.n_test} />
        <Row label="Split" id={`model-${id}-split`} value={split.method} />
      </dl>
      <p data-testid={`model-${id}-licence-status`} role="status">
        {data.licence_verified ? "Licence verified." : "Licence not yet verified: check the source before relying on it."}
      </p>
      {model.assays?.length > 0 && <p data-testid={`model-${id}-assays`}>Assays: {model.assays.join(", ")}</p>}
      {rows.length > 0 && (
        <details open={rows.length <= 12}>
          <summary>Validation metrics</summary>
          <table data-testid={`model-${id}-metrics`}>
            <tbody>
              {rows.map(([name, value]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td data-testid={`model-${id}-metric-${name}`}>{value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </section>
  );
}

export default function ModelInfo() {
  const [models, setModels] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let stopped = false;
    api
      .getModels()
      .then((data) => {
        if (!stopped) setModels(data.models || []);
      })
      .catch((err) => {
        if (!stopped) setError(err);
      });
    return () => {
      stopped = true;
    };
  }, []);

  return (
    <div>
      <h1>Model information</h1>
      <ErrorBanner error={error} />
      {!models && !error && <p>Loading models…</p>}
      {models && models.length === 0 && <p data-testid="models-empty">No models are registered.</p>}
      {models && models.map((model) => <ModelCard key={model.id} model={model} />)}
    </div>
  );
}
