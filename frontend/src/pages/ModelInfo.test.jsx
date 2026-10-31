// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import ModelInfo from "./ModelInfo";

vi.mock("../api/client", () => ({ api: { getModels: vi.fn() } }));
import { api } from "../api/client";

const SOLUBILITY = {
  id: "ml_solubility_xgb_1.0.0",
  tier: "ml",
  name: "ml_solubility_xgb",
  version: "1.0.0",
  task: "solubility",
  unit: "log10(mol/L)",
  target: null,
  trained_at: "2026-10-29T10:00:00Z",
  data: { source: "AqSolDB", citation: "Sorkun et al.", licence_note: "Check the dataset page.", licence_verified: false },
  split: { method: "scaffold_split_bemis_murcko", n_train: 6596, n_test: 1886 },
  metrics: { test: { mae: 0.7539908, r2: 0.7781, n: 1886 } },
  assays: [],
};
const DL = {
  ...SOLUBILITY,
  id: "dl_egfr_gnn_1.0.0",
  tier: "dl",
  name: "dl_egfr_gnn",
  task: "activity",
  unit: "pIC50",
  target: "EGFR",
  data: { source: "ChEMBL", chembl_release: "ChEMBL_37", licence_note: "CC BY-SA 3.0.", licence_verified: true },
};
const TOX21 = { ...SOLUBILITY, id: "ml_tox21_xgb_1.0.0", name: "ml_tox21_xgb", task: "tox21", assays: ["NR-AR", "SR-ARE"] };

beforeEach(() => vi.clearAllMocks());
afterEach(cleanup);

describe("ModelInfo", () => {
  it("shows version, training-data source, licence and validation metrics per model", async () => {
    api.getModels.mockResolvedValue({ models: [SOLUBILITY, DL] });
    render(<ModelInfo />);

    expect((await screen.findByTestId("model-ml_solubility_xgb_1.0.0-version")).textContent).toBe("v1.0.0");
    expect(screen.getByTestId("model-ml_solubility_xgb_1.0.0-source").textContent).toBe("AqSolDB");
    expect(screen.getByTestId("model-ml_solubility_xgb_1.0.0-licence").textContent).toBe("Check the dataset page.");
    expect(screen.getByTestId("model-ml_solubility_xgb_1.0.0-metric-test.mae").textContent).toBe("0.754");
    expect(screen.getByTestId("model-ml_solubility_xgb_1.0.0-metric-test.n").textContent).toBe("1886");
    expect(screen.getByTestId("model-dl_egfr_gnn_1.0.0-target").textContent).toBe("EGFR");
    expect(screen.getByTestId("model-dl_egfr_gnn_1.0.0-release").textContent).toBe("ChEMBL_37");
  });

  it("says plainly when a licence has not been verified", async () => {
    api.getModels.mockResolvedValue({ models: [SOLUBILITY, DL] });
    render(<ModelInfo />);
    const unverified = await screen.findByTestId("model-ml_solubility_xgb_1.0.0-licence-status");
    expect(unverified.textContent).toMatch(/not yet verified/i);
    expect(screen.getByTestId("model-dl_egfr_gnn_1.0.0-licence-status").textContent).toBe("Licence verified.");
  });

  it("lists the assays of a multi-task model", async () => {
    api.getModels.mockResolvedValue({ models: [TOX21] });
    render(<ModelInfo />);
    expect((await screen.findByTestId("model-ml_tox21_xgb_1.0.0-assays")).textContent).toContain("NR-AR, SR-ARE");
  });

  it("shows a message when no models are registered", async () => {
    api.getModels.mockResolvedValue({ models: [] });
    render(<ModelInfo />);
    expect(await screen.findByTestId("models-empty")).toBeTruthy();
  });

  it("shows the API error", async () => {
    api.getModels.mockRejectedValue(new ApiError({ status: 500, code: "INTERNAL", message: "Server error.", requestId: "req-3" }));
    render(<ModelInfo />);
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("Server error.");
    expect(alert.textContent).toContain("req-3");
  });
});
