// pages/Submit.jsx - owner: Abheetha
// Validate a molecule (SMILES or MOL/SDF file), pick a target, then submit it (MSP-35).
import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import ErrorBanner from "../components/ErrorBanner";
import Warnings from "../components/Warnings";

const TIERS = [
  ["ml", "ML"],
  ["dl", "DL"],
  ["qml", "QML"],
];

// skip_reason codes from backend/services/molecule/eligibility.py
const SKIP_TEXT = {
  NO_TARGET_SELECTED: "Select a target to run this tier.",
  UNSUPPORTED_ATOM: "The molecule has atoms this tier does not support.",
  NO_FRAGMENT: "Needs at most nine heavy atoms, all C, N, O or F.",
};
const skipText = (code) => SKIP_TEXT[code] || code || "Not eligible.";

export default function Submit() {
  const navigate = useNavigate();
  const location = useLocation();
  const [smiles, setSmiles] = useState(location.state?.smiles || ""); // "Use this molecule" from Search
  const [file, setFile] = useState(null);
  const [fileKey, setFileKey] = useState(0); // changing the key empties the file input
  const [targetId, setTargetId] = useState("");
  const [targets, setTargets] = useState([]);
  const [targetsError, setTargetsError] = useState(null);
  const [validation, setValidation] = useState(null);
  const [validating, setValidating] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let stopped = false;
    api
      .getTargets()
      .then((data) => {
        if (!stopped) setTargets(data.targets || []);
      })
      .catch((err) => {
        if (!stopped) setTargetsError(err);
      });
    return () => {
      stopped = true;
    };
  }, []);

  // Any change to the input makes an earlier validation out of date.
  const stale = () => {
    setValidation(null);
    setError(null);
  };

  function onSmilesChange(e) {
    setSmiles(e.target.value);
    stale();
  }
  function onTargetChange(e) {
    setTargetId(e.target.value);
    stale(); // DL eligibility depends on the target
  }
  function onFileChange(e) {
    setFile(e.target.files?.[0] || null);
    stale();
  }
  function clearFile() {
    setFile(null);
    setFileKey((n) => n + 1);
    stale();
  }

  async function onValidate(e) {
    e.preventDefault();
    if (!file && !smiles.trim()) return;
    setValidating(true);
    setError(null);
    setValidation(null);
    try {
      const target = targetId || undefined;
      setValidation(file ? await api.validateFile(file, target) : await api.validateSmiles(smiles.trim(), target));
    } catch (err) {
      setError(err);
    } finally {
      setValidating(false);
    }
  }

  async function onSubmit() {
    if (!validation) return;
    setSubmitting(true);
    setError(null);
    try {
      // Submit takes SMILES only, so a file goes in as its standardised SMILES.
      const payload = { smiles: validation.canonical_smiles };
      if (targetId) payload.target_id = targetId;
      const job = await api.submitMolecule(payload);
      navigate(`/jobs/${job.job_id}`);
    } catch (err) {
      setError(err);
      setSubmitting(false);
    }
  }

  const tierErrors = error?.detailsMap?.tiers; // TIER_UNAVAILABLE: { tier: skip_reason }
  const busy = validating || submitting;

  return (
    <div>
      <h1>Submit a molecule</h1>
      <p>
        Have a compound name instead? <Link to="/search" data-testid="submit-search-link">Search PubChem or ChEMBL</Link>.
      </p>

      <form onSubmit={onValidate}>
        <div>
          <label htmlFor="smiles">SMILES</label>
          <br />
          <input
            id="smiles"
            data-testid="submit-smiles"
            type="text"
            value={smiles}
            onChange={onSmilesChange}
            disabled={busy || !!file}
            placeholder="CCO"
            style={{ width: "100%", maxWidth: 480 }}
          />
        </div>

        <div style={{ marginTop: 12 }}>
          <label htmlFor="molfile">Or upload a MOL/SDF file (first record is used)</label>
          <br />
          <input
            key={fileKey}
            id="molfile"
            data-testid="submit-file"
            type="file"
            accept=".mol,.sdf"
            onChange={onFileChange}
            disabled={busy}
          />
          {file && (
            <button type="button" data-testid="submit-file-clear" onClick={clearFile} disabled={busy}>
              Remove file
            </button>
          )}
        </div>

        <div style={{ marginTop: 12 }}>
          <label htmlFor="target">Target</label>
          <br />
          <select id="target" data-testid="submit-target" value={targetId} onChange={onTargetChange} disabled={busy}>
            <option value="">No target (DL tier is skipped)</option>
            {targets.map((t) => (
              <option key={t.id} value={t.id}>
                {t.name}
              </option>
            ))}
          </select>
          {targetsError && (
            <div data-testid="submit-targets-error" role="status">
              Could not load the target list. You can still submit without a target.
            </div>
          )}
        </div>

        <div style={{ marginTop: 12 }}>
          <button type="submit" data-testid="submit-validate" disabled={busy || (!file && !smiles.trim())}>
            {validating ? "Validating…" : "Validate"}
          </button>
        </div>
      </form>

      <ErrorBanner error={error} />
      {tierErrors && (
        <ul data-testid="submit-tier-errors">
          {Object.entries(tierErrors).map(([tier, reason]) => (
            <li key={tier} data-testid={`submit-tier-error-${tier}`}>
              {tier.toUpperCase()}: {skipText(reason)}
            </li>
          ))}
        </ul>
      )}

      {validation && (
        <section data-testid="validate-result" aria-label="Validation result">
          <h2>Validated molecule</h2>
          <dl>
            <dt>Canonical SMILES</dt>
            <dd data-testid="validate-canonical-smiles">{validation.canonical_smiles}</dd>
            <dt>Formula</dt>
            <dd data-testid="validate-formula">{validation.formula}</dd>
            <dt>Heavy atoms</dt>
            <dd data-testid="validate-heavy-atoms">{validation.heavy_atoms}</dd>
            <dt>InChIKey</dt>
            <dd data-testid="validate-inchikey">{validation.inchikey}</dd>
          </dl>
          <Warnings warnings={validation.warnings} />
          <h3>Tiers</h3>
          <ul>
            {TIERS.map(([tier, label]) => {
              const decision = validation.tiers?.[tier];
              if (!decision) return null;
              return (
                <li key={tier} data-testid={`eligibility-${tier}`} data-eligible={String(!!decision.eligible)}>
                  {label}: {decision.eligible ? "Will run" : skipText(decision.skip_reason)}
                </li>
              );
            })}
          </ul>
        </section>
      )}

      <button type="button" data-testid="submit-submit" onClick={onSubmit} disabled={!validation || busy}>
        {submitting ? "Submitting…" : "Submit"}
      </button>
    </div>
  );
}
