// components/TierResult.jsx - owner: Abheetha
// Formats one tier's stored result: a unit beside every value, limitations and warnings visible
// (MSP-37; FR-036, FR-037, NFR-010). Payload shapes: ML in backend/workers/ml/predictor.py (solubility.py,
// tox21.py, alerts.py, domain.py), DL in backend/workers/dl/predictor.py, QML in docs/api/qml-result-payload.md.
// A payload this file does not recognise is shown raw.
import { Fragment } from "react";

const num = (value, digits = 2) => (typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "n/a");

// Energy errors in mHa: ordinary numbers with one decimal, tiny ones in scientific form (0.000188 -> 1.9e-4).
const mha = (value) => {
  if (typeof value !== "number" || !Number.isFinite(value)) return "n/a";
  if (value === 0) return "0";
  return Math.abs(value) >= 0.1 ? value.toFixed(1) : value.toExponential(1);
};
const humanise = (key) => key.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

function Raw({ payload }) {
  return (
    <pre style={{ whiteSpace: "pre-wrap", margin: "4px 0 0", fontSize: 12 }}>{JSON.stringify(payload, null, 2)}</pre>
  );
}

function Value({ label, value, unit, testId }) {
  return (
    <div style={{ margin: "4px 0" }}>
      <span>{label}: </span>
      <strong data-testid={testId}>{unit ? `${value} ${unit}` : value}</strong>
    </div>
  );
}

function Limitation({ text, testId }) {
  if (!text) return null;
  return (
    <p data-testid={testId} style={{ fontSize: 12, color: "#475467", margin: "4px 0" }}>
      {text}
    </p>
  );
}

function Warning({ testId, children }) {
  return (
    <div
      role="note"
      data-testid={testId}
      style={{ border: "1px solid #b54708", background: "#fffaeb", color: "#7a2e0e", padding: "8px 10px", borderRadius: 6, margin: "8px 0" }}
    >
      <strong>Warning: </strong>
      {children}
    </div>
  );
}

const RULE_LABELS = {
  mol_wt: "Molecular weight",
  log_p: "logP",
  h_bond_donors: "H-bond donors",
  h_bond_acceptors: "H-bond acceptors",
  rotatable_bonds: "Rotatable bonds",
  tpsa: "Polar surface area (TPSA)",
};
const RULE_UNITS = { mol_wt: "g/mol", tpsa: "\u00c5\u00b2" };
const FAMILY_LABELS = { solubility: "Solubility model", tox21: "Tox21 models" };

const plain = (value) => (Number.isInteger(value) ? String(value) : num(value));

function DomainBlock({ domain }) {
  const details = (
    <ul data-testid="ml-domain-details" style={{ margin: "4px 0 0 18px", fontSize: 12 }}>
      {Object.entries(domain.families || {}).map(([name, family]) => (
        <li key={name} data-testid={`ml-domain-${name}`}>
          {FAMILY_LABELS[name] || name}:{" "}
          {family.available === false
            ? "not checked (reference data unavailable)"
            : `highest Tanimoto similarity to the training molecules ${num(family.max_tanimoto)} (minimum ${num(family.threshold)})`}
          {family.descriptors_outside_range?.length > 0 && `; outside the training range: ${family.descriptors_outside_range.join(", ")}`}
        </li>
      ))}
    </ul>
  );
  if (domain.in_domain === false) {
    return (
      <Warning testId="ml-out-of-domain">
        <strong>Outside the applicability domain.</strong> {domain.warning}
        {details}
      </Warning>
    );
  }
  if (domain.in_domain === null || domain.in_domain === undefined) {
    return (
      <Warning testId="ml-domain-unchecked">
        <strong>Applicability domain not checked.</strong> {domain.warning}
        {details}
      </Warning>
    );
  }
  return (
    <div data-testid="ml-in-domain" style={{ fontSize: 12 }}>
      Inside the applicability domain.
      {details}
    </div>
  );
}

function SolubilityBlock({ solubility }) {
  const { interval } = solubility;
  return (
    <div data-testid="ml-solubility" style={{ margin: "8px 0" }}>
      <Value label="Solubility (logS)" value={num(solubility.logs)} unit={solubility.unit} testId="ml-solubility-value" />
      {interval && (
        <div data-testid="ml-solubility-interval" style={{ fontSize: 12 }}>
          {num(interval.level * 100, 0)}% prediction interval: {num(interval.low)} to {num(interval.high)} {solubility.unit}
        </div>
      )}
      <Limitation text={solubility.limitation} testId="ml-solubility-limitation" />
    </div>
  );
}

function Tox21Block({ tox21 }) {
  return (
    <div data-testid="ml-tox21" style={{ margin: "8px 0" }}>
      <strong>Tox21 assay signals</strong>
      <table style={{ fontSize: 12, marginTop: 4 }}>
        <thead>
          <tr>
            <th align="left">Assay</th>
            <th align="left">Signal ({tox21.unit})</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(tox21.signals || {}).map(([assay, signal]) => (
            <tr key={assay}>
              <td>{assay}</td>
              <td data-testid={`ml-tox21-${assay}`}>
                {signal.probability === null || signal.probability === undefined ? "unavailable" : num(signal.probability)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <Limitation text={tox21.limitation} testId="ml-tox21-limitation" />
    </div>
  );
}

function AlertsBlock({ alerts }) {
  const list = (names) => (names?.length ? names.join(", ") : "none");
  return (
    <div data-testid="ml-alerts" style={{ margin: "8px 0" }}>
      <strong>Structural alerts</strong>
      <div data-testid="ml-alerts-pains">PAINS ({alerts.n_pains ?? 0}): {list(alerts.pains)}</div>
      <div data-testid="ml-alerts-brenk">Brenk ({alerts.n_brenk ?? 0}): {list(alerts.brenk)}</div>
      <Limitation text={alerts.limitation} testId="ml-alerts-limitation" />
    </div>
  );
}

function RuleTable({ title, rule, name }) {
  return (
    <div data-testid={`ml-rules-${name}`} style={{ margin: "4px 0" }}>
      <div>
        {title}: <span data-testid={`ml-rules-${name}-violations`}>{rule.violations} violation{rule.violations === 1 ? "" : "s"}</span>
      </div>
      <table style={{ fontSize: 12 }}>
        <tbody>
          {Object.entries(rule.values || {}).map(([key, value]) => {
            const unit = RULE_UNITS[key] ? ` ${RULE_UNITS[key]}` : "";
            const failed = rule.failed?.includes(key);
            return (
              <tr key={key} data-testid={`ml-rules-${name}-${key}`} data-failed={String(Boolean(failed))}>
                <td>{RULE_LABELS[key] || key}</td>
                <td>
                  {plain(value)}
                  {unit}
                </td>
                <td>
                  limit {plain(rule.limits?.[key])}
                  {unit}
                </td>
                <td>{failed ? "above the limit" : ""}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function RulesBlock({ rules }) {
  return (
    <div data-testid="ml-rules" style={{ margin: "8px 0" }}>
      <strong>Property rules</strong>
      {rules.lipinski && <RuleTable title="Lipinski rule of five" rule={rules.lipinski} name="lipinski" />}
      {rules.veber && <RuleTable title="Veber" rule={rules.veber} name="veber" />}
      <Limitation text={rules.limitation} testId="ml-rules-limitation" />
    </div>
  );
}

function MlResult({ payload }) {
  return (
    <div data-testid="ml-block">
      {payload.domain && <DomainBlock domain={payload.domain} />}
      {payload.solubility && <SolubilityBlock solubility={payload.solubility} />}
      {payload.tox21 && <Tox21Block tox21={payload.tox21} />}
      {payload.alerts && <AlertsBlock alerts={payload.alerts} />}
      {payload.rules && <RulesBlock rules={payload.rules} />}
      {payload.domain?.limitation && <Limitation text={payload.domain.limitation} testId="ml-domain-limitation" />}
    </div>
  );
}

function DlResult({ payload }) {
  const { activity, baseline, target } = payload;
  return (
    <div data-testid="dl-block">
      {target?.name && <div data-testid="dl-target">Target: {target.name}</div>}
      <Value label="Predicted activity (ensemble mean)" value={num(activity.mean)} unit={activity.unit} testId="dl-activity-mean" />
      <Value label="Ensemble spread (SD)" value={num(activity.sd)} unit={activity.unit} testId="dl-activity-sd" />
      {activity.n_models != null && (
        <div data-testid="dl-activity-models" style={{ fontSize: 12 }}>
          Ensemble of {activity.n_models} models
        </div>
      )}
      <Limitation text={activity.limitation} testId="dl-activity-limitation" />
      {baseline && (
        <div style={{ marginTop: 8 }}>
          <Value label="Baseline model" value={num(baseline.value)} unit={baseline.unit} testId="dl-baseline-value" />
          {baseline.algorithm && (
            <div data-testid="dl-baseline-algorithm" style={{ fontSize: 12 }}>
              {baseline.algorithm}
            </div>
          )}
          <Limitation text={baseline.limitation} testId="dl-baseline-limitation" />
        </div>
      )}
    </div>
  );
}

function QmlResult({ payload }) {
  const { electronic_stability: stability, convergence, backend, active_space: space } = payload;
  const refs = payload.classical_references;
  const run = payload.run;
  const stub = payload.is_stub === true || backend?.is_stub === true;
  const label = stability.metric === "vqe_excitation_gap" ? "VQE excitation gap" : "HOMO-LUMO gap";
  const usesFci = convergence?.error_vs_fci_mha !== undefined && convergence?.error_vs_fci_mha !== null;
  return (
    <div data-testid="qml-block">
      {stub && (
        <Warning testId="qml-stub">
          Illustrative stub value. No quantum calculation was run, and the same number is returned for every
          molecule. Do not use it to compare or rank molecules.
        </Warning>
      )}
      <Value label={label} value={num(stability.value)} unit={stability.unit} testId="qml-value" />
      {convergence?.converged === false && (
        <Warning testId="qml-not-converged">
          <strong>Not converged.</strong> {stability.warning || "The value is shown, but treat it with caution."}{" "}
          {usesFci
            ? `The error against full CI is ${mha(convergence.error_vs_fci_mha)} mHa, above the ${mha(convergence.threshold_mha)} mHa threshold.`
            : `The last energy change is ${mha(convergence.final_change_mha)} mHa, above the ${mha(convergence.threshold_mha)} mHa threshold.`}
        </Warning>
      )}
      {payload.excluded_from_triage === true && (
        <div data-testid="qml-excluded" style={{ fontSize: 12, margin: "4px 0" }}>
          Not used in the triage summary.
        </div>
      )}
      {convergence?.converged === true && (
        <div data-testid="qml-converged" style={{ fontSize: 12 }}>
          Converged after {convergence.iterations} iterations (final change {mha(convergence.final_change_mha)} mHa
          {usesFci ? `; error against full CI ${mha(convergence.error_vs_fci_mha)} mHa` : ""}).
        </div>
      )}
      <Limitation text={stability.limitation} testId="qml-limitation" />
      {refs && (
        <div data-testid="qml-references" style={{ marginTop: 8 }}>
          <Value label="Hartree-Fock gap (classical reference)" value={num(refs.hf_gap_ev)} unit={refs.unit} testId="qml-ref-hf" />
          <Value label="Full-CI gap (classical reference)" value={num(refs.fci_gap_ev)} unit={refs.unit} testId="qml-ref-fci" />
          <Limitation text={refs.limitation} testId="qml-ref-limitation" />
        </div>
      )}
      {payload.energies?.gap_error_ev !== undefined && payload.energies?.gap_error_ev !== null && (
        <div data-testid="qml-gap-error" style={{ fontSize: 12 }}>
          Gap error against full CI: {num(payload.energies.gap_error_ev, 3)} eV
        </div>
      )}
      {backend && (
        <div data-testid="qml-backend" style={{ fontSize: 12 }}>
          Backend: {backend.name} {backend.version}
        </div>
      )}
      {space && (
        <div data-testid="qml-active-space" style={{ fontSize: 12 }}>
          Active space: {space.electrons} electrons in {space.orbitals} orbitals
        </div>
      )}
      {run && (
        <details data-testid="qml-run" style={{ fontSize: 12, marginTop: 4 }}>
          <summary>How it was calculated</summary>
          <dl>
            {Object.entries(run).map(([key, value]) => (
              <Fragment key={key}>
                <dt>{humanise(key)}</dt>
                <dd data-testid={`qml-run-${key}`}>{key === "seconds" ? `${num(value, 1)} s` : String(value)}</dd>
              </Fragment>
            ))}
          </dl>
        </details>
      )}
    </div>
  );
}

export default function TierResult({ tier, payload }) {
  if (tier === "ml" && (payload?.solubility || payload?.tox21)) return <MlResult payload={payload} />;
  if (tier === "dl" && payload?.activity) return <DlResult payload={payload} />;
  if (tier === "qml" && payload?.electronic_stability) return <QmlResult payload={payload} />;
  return <Raw payload={payload} />;
}
