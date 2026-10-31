// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import JobStatus from "./JobStatus";
import TierResult from "./TierResult";

afterEach(cleanup);

const DL = {
  tier: "dl",
  target: { id: "EGFR", name: "Epidermal growth factor receptor (EGFR)", chembl_id: "CHEMBL203" },
  activity: {
    mean: 6.1234, sd: 0.3156, n_models: 5, unit: "pIC50", model_version: "dl_egfr_gnn@1.0.0",
    limitation: "Predicted by an ensemble of five graph neural networks.",
  },
  baseline: { value: 5.9, unit: "pIC50", algorithm: "gradient-boosted trees (XGBoost)", limitation: "Shown only for comparison." },
};
const QML = (overrides = {}) => ({
  tier: "qml",
  is_stub: false,
  electronic_stability: { metric: "homo_lumo_gap", value: 4.5, unit: "eV", limitation: "An estimate.", model_version: "qml_x@1.0.0" },
  convergence: { converged: true, iterations: 12, final_change_mha: 0.9, threshold_mha: 1.6 },
  backend: { name: "vqe", version: "1.0.0", is_stub: false },
  active_space: { electrons: 4, orbitals: 4 },
  ...overrides,
});

describe("TierResult: DL", () => {
  it("puts the unit beside every value and shows both limitations", () => {
    render(<TierResult tier="dl" payload={DL} />);
    expect(screen.getByTestId("dl-activity-mean").textContent).toBe("6.12 pIC50");
    expect(screen.getByTestId("dl-activity-sd").textContent).toBe("0.32 pIC50");
    expect(screen.getByTestId("dl-baseline-value").textContent).toBe("5.90 pIC50");
    expect(screen.getByTestId("dl-activity-limitation").textContent).toContain("five graph neural networks");
    expect(screen.getByTestId("dl-baseline-limitation").textContent).toContain("comparison");
    expect(screen.getByTestId("dl-target").textContent).toContain("EGFR");
  });

  it("falls back to the raw payload when it does not have the expected shape", () => {
    render(<TierResult tier="dl" payload={{ pIC50: 6.1 }} />);
    expect(screen.queryByTestId("dl-block")).toBeNull();
    expect(document.body.textContent).toContain('"pIC50": 6.1');
  });
});

describe("TierResult: QML", () => {
  it("shows the value in eV and a converged note without warnings", () => {
    render(<TierResult tier="qml" payload={QML()} />);
    expect(screen.getByTestId("qml-value").textContent).toBe("4.50 eV");
    expect(screen.getByTestId("qml-converged").textContent).toContain("12 iterations");
    expect(screen.queryByTestId("qml-not-converged")).toBeNull();
    expect(screen.queryByTestId("qml-stub")).toBeNull();
  });

  it("flags a result that did not converge, and still shows the value", () => {
    const payload = QML({ convergence: { converged: false, iterations: 40, final_change_mha: 2.4, threshold_mha: 1.6 } });
    render(<TierResult tier="qml" payload={payload} />);
    expect(screen.getByTestId("qml-not-converged").textContent).toMatch(/Not converged/);
    expect(screen.getByTestId("qml-not-converged").textContent).toContain("2.4 mHa");
    expect(screen.getByTestId("qml-value").textContent).toBe("4.50 eV");
    expect(screen.queryByTestId("qml-converged")).toBeNull();
  });

  it("marks a stub value as illustrative", () => {
    render(<TierResult tier="qml" payload={QML({ is_stub: true, backend: { name: "stub", version: "1.0.0", is_stub: true } })} />);
    expect(screen.getByTestId("qml-stub").textContent).toMatch(/Illustrative stub value/);
    expect(screen.getByTestId("qml-stub").textContent).toMatch(/rank/);
  });

  it("treats a backend-level stub flag as a stub too", () => {
    render(<TierResult tier="qml" payload={QML({ backend: { name: "stub", version: "1.0.0", is_stub: true } })} />);
    expect(screen.getByTestId("qml-stub")).toBeTruthy();
  });
});

describe("JobStatus with formatted results and the triage summary", () => {
  const job = (summary) => ({
    job_id: "j-1",
    status: "complete",
    summary,
    tasks: [{ id: "t1", tier: "dl", status: "done", result: { model_version: "dl_egfr_gnn@1.0.0", payload: DL } }],
  });

  it("shows the formatted DL block inside the tier row", () => {
    render(<JobStatus job={job(null)} />);
    expect(screen.getByTestId("tier-dl-result").textContent).toContain("6.12 pIC50");
  });

  it("says the triage summary is not available while jobs.summary is empty", () => {
    render(<JobStatus job={job(null)} />);
    expect(screen.getByTestId("triage-pending")).toBeTruthy();
    expect(screen.queryByTestId("triage-summary")).toBeNull();
  });

  it("shows the summary entries once the aggregator has written them", () => {
    render(<JobStatus job={job({ developability: "pass", confidence: "low" })} />);
    expect(screen.getByTestId("triage-developability").textContent).toBe("pass");
    expect(screen.getByTestId("triage-confidence").textContent).toBe("low");
    expect(screen.queryByTestId("triage-pending")).toBeNull();
  });
});

const DOMAIN_OK = {
  in_domain: true,
  warning: null,
  families: {
    solubility: { available: true, in_domain: true, max_tanimoto: 0.52, threshold: 0.3, descriptors_outside_range: [] },
    tox21: { available: true, in_domain: true, max_tanimoto: 0.47, threshold: 0.3, descriptors_outside_range: [] },
  },
};
const ML = (domain = DOMAIN_OK, extra = {}) => ({
  tier: "ml",
  solubility: {
    logs: -2.5, unit: "log10(mol/L)", limitation: "An indicative value, not a measurement.",
    interval: { low: -3.4, high: -1.6, level: 0.9 },
  },
  tox21: {
    unit: "calibrated probability (0 to 1)", limitation: "Not a toxicity assessment.",
    signals: {
      "NR-AR": { probability: 0.1234, label: "assay_signal" },
      "SR-p53": { probability: null, status: "unavailable", label: "assay_signal" },
    },
  },
  alerts: { pains: [], brenk: ["Aliphatic long chain"], n_pains: 0, n_brenk: 1, limitation: "Alerts describe, they do not judge." },
  rules: {
    lipinski: {
      values: { mol_wt: 612.3, log_p: 3.2, h_bond_donors: 2, h_bond_acceptors: 7 },
      limits: { mol_wt: 500, log_p: 5, h_bond_donors: 5, h_bond_acceptors: 10 },
      failed: ["mol_wt"], violations: 1,
    },
    veber: { values: { rotatable_bonds: 4, tpsa: 90.5 }, limits: { rotatable_bonds: 10, tpsa: 140 }, failed: [], violations: 0 },
    limitation: "Rules of thumb only.",
  },
  domain,
  ...extra,
});

describe("TierResult: ML", () => {
  it("puts units beside the solubility value and its interval", () => {
    render(<TierResult tier="ml" payload={ML()} />);
    expect(screen.getByTestId("ml-solubility-value").textContent).toBe("-2.50 log10(mol/L)");
    expect(screen.getByTestId("ml-solubility-interval").textContent).toBe("90% prediction interval: -3.40 to -1.60 log10(mol/L)");
    expect(screen.getByTestId("ml-solubility-limitation").textContent).toContain("not a measurement");
  });

  it("lists the Tox21 signals with their unit and shows an unavailable assay as unavailable", () => {
    render(<TierResult tier="ml" payload={ML()} />);
    expect(screen.getByTestId("ml-tox21-NR-AR").textContent).toBe("0.12");
    expect(screen.getByTestId("ml-tox21-SR-p53").textContent).toBe("unavailable");
    expect(screen.getByTestId("ml-tox21").textContent).toContain("calibrated probability (0 to 1)");
    expect(screen.getByTestId("ml-tox21-limitation").textContent).toContain("Not a toxicity assessment");
  });

  it("shows the alerts and the rule violations with units, limits and the failed rule", () => {
    render(<TierResult tier="ml" payload={ML()} />);
    expect(screen.getByTestId("ml-alerts-pains").textContent).toContain("none");
    expect(screen.getByTestId("ml-alerts-brenk").textContent).toContain("Aliphatic long chain");
    expect(screen.getByTestId("ml-rules-lipinski-violations").textContent).toBe("1 violation");
    const weight = screen.getByTestId("ml-rules-lipinski-mol_wt");
    expect(weight.textContent).toContain("612.30 g/mol");
    expect(weight.textContent).toContain("limit 500 g/mol");
    expect(weight.getAttribute("data-failed")).toBe("true");
    expect(screen.getByTestId("ml-rules-lipinski-log_p").getAttribute("data-failed")).toBe("false");
    expect(screen.getByTestId("ml-rules-veber-tpsa").textContent).toContain("90.50 \u00c5\u00b2");
  });

  it("flags a molecule outside the applicability domain visibly, with the reason", () => {
    const domain = {
      in_domain: false,
      warning: "The molecule is unlike the training data; treat the values with caution.",
      families: {
        solubility: { available: true, in_domain: false, max_tanimoto: 0.21, threshold: 0.3, descriptors_outside_range: ["MolWt", "TPSA"] },
        tox21: { available: true, in_domain: true, max_tanimoto: 0.4, threshold: 0.3, descriptors_outside_range: [] },
      },
    };
    render(<TierResult tier="ml" payload={ML(domain)} />);
    const flag = screen.getByTestId("ml-out-of-domain");
    expect(flag.textContent).toContain("Outside the applicability domain");
    expect(flag.textContent).toContain("unlike the training data");
    expect(screen.getByTestId("ml-domain-solubility").textContent).toContain("0.21");
    expect(screen.getByTestId("ml-domain-solubility").textContent).toContain("MolWt, TPSA");
    expect(screen.queryByTestId("ml-in-domain")).toBeNull();
    expect(screen.getByTestId("ml-solubility-value")).toBeTruthy();
  });

  it("says plainly when the domain could not be checked, and never calls it inside", () => {
    const domain = {
      in_domain: null,
      warning: "The domain check was not available.",
      families: { solubility: { available: false, in_domain: null, max_tanimoto: null, threshold: 0.3, descriptors_outside_range: [] } },
    };
    render(<TierResult tier="ml" payload={ML(domain)} />);
    expect(screen.getByTestId("ml-domain-unchecked").textContent).toContain("not checked");
    expect(screen.queryByTestId("ml-in-domain")).toBeNull();
    expect(screen.queryByTestId("ml-out-of-domain")).toBeNull();
  });

  it("shows no warning for a molecule inside the domain", () => {
    render(<TierResult tier="ml" payload={ML()} />);
    expect(screen.getByTestId("ml-in-domain").textContent).toContain("Inside the applicability domain");
    expect(screen.queryByTestId("ml-out-of-domain")).toBeNull();
  });

  it("falls back to the raw payload when it has no ML blocks", () => {
    render(<TierResult tier="ml" payload={{ logS: -2.5 }} />);
    expect(screen.queryByTestId("ml-block")).toBeNull();
    expect(document.body.textContent).toContain('"logS": -2.5');
  });
});

const VQE = (overrides = {}) => ({
  tier: "qml",
  is_stub: false,
  electronic_stability: {
    metric: "vqe_excitation_gap", value: 13.271136, unit: "eV",
    limitation: "Estimated by a quantum-chemistry calculation; not a measurement.", model_version: "qml_vqe@1.0.0",
  },
  convergence: { converged: true, iterations: 5, final_change_mha: 6e-6, threshold_mha: 1.6, error_vs_fci_mha: 0.000188 },
  backend: { name: "vqe", version: "1.0.0", is_stub: false },
  active_space: { electrons: 4, orbitals: 4 },
  excluded_from_triage: false,
  classical_references: { hf_gap_ev: 25.30510216, fci_gap_ev: 13.27113558, unit: "eV", limitation: "Classical reference values." },
  run: { mapping: "jordan_wigner", ansatz: "UCCSD (from the Hartree-Fock state)", qubits: 8, seconds: 11.7 },
  energies: { gap_error_ev: 0.0 },
  ...overrides,
});

describe("TierResult: QML real backend", () => {
  it("shows the VQE gap in eV beside the classical reference gaps, with no stub warning", () => {
    render(<TierResult tier="qml" payload={VQE()} />);
    expect(screen.getByTestId("qml-value").textContent).toBe("13.27 eV");
    expect(screen.getByTestId("qml-ref-hf").textContent).toBe("25.31 eV");
    expect(screen.getByTestId("qml-ref-fci").textContent).toBe("13.27 eV");
    expect(screen.getByTestId("qml-ref-limitation").textContent).toContain("Classical reference");
    expect(screen.getByTestId("qml-block").textContent).toContain("VQE excitation gap");
    expect(screen.queryByTestId("qml-stub")).toBeNull();
    expect(screen.queryByTestId("qml-excluded")).toBeNull();
  });

  it("reports the error against full CI in scientific form when it is tiny", () => {
    render(<TierResult tier="qml" payload={VQE()} />);
    expect(screen.getByTestId("qml-converged").textContent).toContain("5 iterations");
    expect(screen.getByTestId("qml-converged").textContent).toContain("error against full CI 1.9e-4 mHa");
  });

  it("flags a result that did not converge, shows the worker's warning and keeps the value", () => {
    const payload = VQE({
      electronic_stability: {
        metric: "vqe_excitation_gap", value: 12.9, unit: "eV", limitation: "An estimate.",
        warning: "The optimisation did not converge; treat this value with caution.",
      },
      convergence: { converged: false, iterations: 60, final_change_mha: 0.2, threshold_mha: 1.6, error_vs_fci_mha: 3.2 },
      excluded_from_triage: true,
    });
    render(<TierResult tier="qml" payload={payload} />);
    const warning = screen.getByTestId("qml-not-converged").textContent;
    expect(warning).toContain("Not converged");
    expect(warning).toContain("treat this value with caution");
    expect(warning).toContain("error against full CI is 3.2 mHa");
    expect(screen.getByTestId("qml-value").textContent).toBe("12.90 eV");
    expect(screen.getByTestId("qml-excluded").textContent).toContain("Not used in the triage summary");
    expect(screen.queryByTestId("qml-converged")).toBeNull();
  });

  it("lists how the result was calculated", () => {
    render(<TierResult tier="qml" payload={VQE()} />);
    expect(screen.getByTestId("qml-run-mapping").textContent).toBe("jordan_wigner");
    expect(screen.getByTestId("qml-run-qubits").textContent).toBe("8");
    expect(screen.getByTestId("qml-run-seconds").textContent).toBe("11.7 s");
    expect(screen.getByTestId("qml-gap-error").textContent).toContain("0.000 eV");
  });
});
