// pages/Search.jsx - owner: Abheetha
// Search PubChem or ChEMBL by compound name and hand a hit to the Submit page (MSP-35).
// Direct SMILES entry lives on /submit and does not depend on this page or these services.
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import ErrorBanner from "../components/ErrorBanner";

const SOURCES = [
  ["pubchem", "PubChem"],
  ["chembl", "ChEMBL"],
];

export default function Search() {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [source, setSource] = useState("pubchem");
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function onSearch(e) {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const data = await api.searchMolecules(q, source);
      setResults(data.results || []);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }

  function pickMolecule(hit) {
    navigate("/submit", { state: { smiles: hit.smiles } });
  }

  // A service problem (or a dropped connection) is not the user's fault; point to the way around it.
  const serviceDown = error && (error.status === 0 || error.status >= 500);

  return (
    <div>
      <h1>Search molecules</h1>
      <form onSubmit={onSearch}>
        <label htmlFor="query">Compound name</label>{" "}
        <input
          id="query"
          data-testid="search-query"
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={loading}
          maxLength={200}
          placeholder="aspirin"
        />{" "}
        <label htmlFor="source">Source</label>{" "}
        <select id="source" data-testid="search-source" value={source} onChange={(e) => setSource(e.target.value)} disabled={loading}>
          {SOURCES.map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>{" "}
        <button type="submit" data-testid="search-submit" disabled={loading || !query.trim()}>
          {loading ? "Searching…" : "Search"}
        </button>
      </form>

      <ErrorBanner error={error} />
      {serviceDown && (
        <p data-testid="search-fallback">
          Search is unavailable right now. You can still{" "}
          <Link to="/submit" data-testid="search-fallback-link">
            enter a SMILES directly
          </Link>
          .
        </p>
      )}

      {results && (
        <ul data-testid="search-results" style={{ listStyle: "none", padding: 0 }}>
          {results.map((hit, i) => (
            <li key={`${hit.source}-${hit.id ?? i}`} data-testid={`result-${i}`} style={{ display: "flex", gap: 12, margin: "12px 0" }}>
              {hit.structure_image_url && (
                <img
                  src={hit.structure_image_url}
                  alt={`Structure of ${hit.name || hit.id || "compound"}`}
                  width={96}
                  height={96}
                  loading="lazy"
                />
              )}
              <div>
                <strong data-testid={`result-${i}-name`}>{hit.name || "Unnamed compound"}</strong>
                <div data-testid={`result-${i}-id`}>
                  {hit.source === "chembl" ? "ChEMBL" : "PubChem"} {hit.id}
                </div>
                {hit.formula && <div data-testid={`result-${i}-formula`}>{hit.formula}</div>}
                {hit.molecular_weight != null && <div data-testid={`result-${i}-weight`}>MW {hit.molecular_weight}</div>}
                <div data-testid={`result-${i}-smiles`} style={{ wordBreak: "break-all", fontFamily: "monospace" }}>
                  {hit.smiles || "No structure available"}
                </div>
                <button type="button" data-testid={`result-${i}-use`} disabled={!hit.smiles} onClick={() => pickMolecule(hit)}>
                  Use this molecule
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
