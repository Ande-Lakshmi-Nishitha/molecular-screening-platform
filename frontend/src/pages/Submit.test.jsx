// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import Submit from "./Submit";

vi.mock("../api/client", () => ({
  api: { getTargets: vi.fn(), validateSmiles: vi.fn(), validateFile: vi.fn(), submitMolecule: vi.fn() },
}));
import { api } from "../api/client";

const TARGETS = {
  targets: [
    { id: "EGFR", name: "Epidermal growth factor receptor (EGFR)", chembl_id: "CHEMBL203", unit: "pIC50" },
    { id: "BACE1", name: "Beta-secretase 1 (BACE1)", chembl_id: "CHEMBL4822", unit: "pIC50" },
    { id: "HIV1_PROTEASE", name: "HIV-1 protease", chembl_id: "CHEMBL243", unit: "pKi" },
  ],
};
const VALID = {
  canonical_smiles: "CCO",
  inchikey: "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
  formula: "C2H6O",
  heavy_atoms: 3,
  warnings: ["The largest fragment was kept."],
  tiers: {
    ml: { eligible: true, skip_reason: null },
    dl: { eligible: false, skip_reason: "NO_TARGET_SELECTED" },
    qml: { eligible: true, skip_reason: null },
  },
};

function renderSubmit(state) {
  return render(
    <MemoryRouter initialEntries={[{ pathname: "/submit", state }]}>
      <Routes>
        <Route path="/submit" element={<Submit />} />
        <Route path="/jobs/:id" element={<div data-testid="job-page" />} />
      </Routes>
    </MemoryRouter>,
  );
}
const typeSmiles = (value) => fireEvent.change(screen.getByTestId("submit-smiles"), { target: { value } });
const click = (id) => fireEvent.click(screen.getByTestId(id));

beforeEach(() => {
  vi.clearAllMocks();
  api.getTargets.mockResolvedValue(TARGETS);
});
afterEach(cleanup);

describe("Submit", () => {
  it("fills the target selector from /api/targets", async () => {
    renderSubmit();
    await screen.findByRole("option", { name: /EGFR/ });
    expect(screen.getByTestId("submit-target").options.length).toBe(4); // "no target" plus three
  });

  it("starts with Validate and Submit disabled", async () => {
    renderSubmit();
    await screen.findByRole("option", { name: /EGFR/ });
    expect(screen.getByTestId("submit-validate").disabled).toBe(true);
    expect(screen.getByTestId("submit-submit").disabled).toBe(true);
  });

  it("shows the canonical SMILES, warnings and per-tier eligibility after validating", async () => {
    api.validateSmiles.mockResolvedValue(VALID);
    renderSubmit();
    typeSmiles(" OCC ");
    click("submit-validate");

    expect((await screen.findByTestId("validate-canonical-smiles")).textContent).toBe("CCO");
    expect(api.validateSmiles).toHaveBeenCalledWith("OCC", undefined);
    expect(screen.getByTestId("validation-warning-0").textContent).toContain("largest fragment");
    expect(screen.getByTestId("eligibility-ml").textContent).toContain("Will run");
    expect(screen.getByTestId("eligibility-dl").textContent).toContain("Select a target");
    expect(screen.getByTestId("submit-submit").disabled).toBe(false);
  });

  it("submits the canonical SMILES with the chosen target and opens the job tracker", async () => {
    api.validateSmiles.mockResolvedValue(VALID);
    api.submitMolecule.mockResolvedValue({ job_id: "j-1", molecule: {}, tasks: [] });
    renderSubmit();
    await screen.findByRole("option", { name: /EGFR/ });
    fireEvent.change(screen.getByTestId("submit-target"), { target: { value: "EGFR" } });
    typeSmiles("OCC");
    click("submit-validate");
    await screen.findByTestId("validate-result");
    expect(api.validateSmiles).toHaveBeenCalledWith("OCC", "EGFR");

    click("submit-submit");
    await screen.findByTestId("job-page");
    expect(api.submitMolecule).toHaveBeenCalledWith({ smiles: "CCO", target_id: "EGFR" });
  });

  it("shows a validation error, keeps Submit disabled and shows no result", async () => {
    api.validateSmiles.mockRejectedValue(
      new ApiError({ status: 422, code: "VALIDATION_ERROR", message: "The SMILES could not be parsed.", requestId: "req-7" }),
    );
    renderSubmit();
    typeSmiles("C(C");
    click("submit-validate");

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("could not be parsed");
    expect(alert.textContent).toContain("req-7");
    expect(screen.queryByTestId("validate-result")).toBeNull();
    expect(screen.getByTestId("submit-submit").disabled).toBe(true);
  });

  it("drops an earlier validation when the target changes", async () => {
    api.validateSmiles.mockResolvedValue(VALID);
    renderSubmit();
    await screen.findByRole("option", { name: /EGFR/ });
    typeSmiles("OCC");
    click("submit-validate");
    await screen.findByTestId("validate-result");

    fireEvent.change(screen.getByTestId("submit-target"), { target: { value: "BACE1" } });
    expect(screen.queryByTestId("validate-result")).toBeNull();
    expect(screen.getByTestId("submit-submit").disabled).toBe(true);
  });

  it("validates an uploaded MOL/SDF file instead of the SMILES box", async () => {
    api.validateFile.mockResolvedValue({ ...VALID, warnings: [] });
    renderSubmit();
    const file = new File(["mol"], "ethanol.sdf");
    fireEvent.change(screen.getByTestId("submit-file"), { target: { files: [file] } });
    expect(screen.getByTestId("submit-smiles").disabled).toBe(true);
    click("submit-validate");

    await screen.findByTestId("validate-result");
    expect(api.validateFile).toHaveBeenCalledWith(file, undefined);
    expect(api.validateSmiles).not.toHaveBeenCalled();
  });

  it("names the tiers that cannot run when submit answers TIER_UNAVAILABLE", async () => {
    api.validateSmiles.mockResolvedValue(VALID);
    api.submitMolecule.mockRejectedValue(
      new ApiError({
        status: 422,
        code: "TIER_UNAVAILABLE",
        message: "A requested tier cannot run for this molecule.",
        detailsMap: { tiers: { dl: "NO_TARGET_SELECTED" } },
      }),
    );
    renderSubmit();
    typeSmiles("OCC");
    click("submit-validate");
    await screen.findByTestId("validate-result");
    click("submit-submit");

    expect((await screen.findByTestId("submit-tier-error-dl")).textContent).toContain("Select a target");
    expect(screen.queryByTestId("job-page")).toBeNull();
    expect(screen.getByTestId("submit-submit").disabled).toBe(false); // the user can fix it and retry
  });

  it("prefills the SMILES sent from the Search page", async () => {
    renderSubmit({ smiles: "c1ccccc1" });
    await screen.findByRole("option", { name: /EGFR/ });
    expect(screen.getByTestId("submit-smiles").value).toBe("c1ccccc1");
  });

  it("still works when the target list cannot be loaded", async () => {
    api.getTargets.mockRejectedValue(new ApiError({ status: 500, code: "INTERNAL", message: "Server error." }));
    api.validateSmiles.mockResolvedValue(VALID);
    renderSubmit();
    expect((await screen.findByTestId("submit-targets-error")).textContent).toContain("without a target");
    typeSmiles("OCC");
    click("submit-validate");
    await screen.findByTestId("validate-result");
  });
});
