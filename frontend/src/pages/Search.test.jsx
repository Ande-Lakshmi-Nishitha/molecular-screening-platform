// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import Search from "./Search";

vi.mock("../api/client", () => ({ api: { searchMolecules: vi.fn() } }));
import { api } from "../api/client";

const ASPIRIN = {
  source: "pubchem",
  id: "2244",
  name: "2-acetyloxybenzoic acid",
  smiles: "CC(=O)OC1=CC=CC=C1C(=O)O",
  inchi: "InChI=1S/C9H8O4",
  inchi_key: "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
  formula: "C9H8O4",
  molecular_weight: 180.16,
  structure_image_url: "https://example.test/2244.png",
};
const NO_STRUCTURE = { ...ASPIRIN, id: "9", name: null, smiles: null, structure_image_url: null };
const response = (results, source = "pubchem") => ({ source, query: "aspirin", results, cached: false });

function SubmitStub() {
  const { state } = useLocation();
  return <div data-testid="submit-stub">{state?.smiles}</div>;
}
function renderSearch() {
  return render(
    <MemoryRouter initialEntries={["/search"]}>
      <Routes>
        <Route path="/search" element={<Search />} />
        <Route path="/submit" element={<SubmitStub />} />
      </Routes>
    </MemoryRouter>,
  );
}
const search = (text) => {
  fireEvent.change(screen.getByTestId("search-query"), { target: { value: text } });
  fireEvent.click(screen.getByTestId("search-submit"));
};

beforeEach(() => vi.clearAllMocks());
afterEach(cleanup);

describe("Search", () => {
  it("does not search while the query is empty", () => {
    renderSearch();
    expect(screen.getByTestId("search-submit").disabled).toBe(true);
    fireEvent.change(screen.getByTestId("search-query"), { target: { value: "   " } });
    expect(screen.getByTestId("search-submit").disabled).toBe(true);
    expect(api.searchMolecules).not.toHaveBeenCalled();
  });

  it("searches PubChem by default and lists the hits", async () => {
    api.searchMolecules.mockResolvedValue(response([ASPIRIN]));
    renderSearch();
    search("  aspirin ");

    expect((await screen.findByTestId("result-0-name")).textContent).toBe("2-acetyloxybenzoic acid");
    expect(api.searchMolecules).toHaveBeenCalledWith("aspirin", "pubchem");
    expect(screen.getByTestId("result-0-formula").textContent).toBe("C9H8O4");
    expect(screen.getByTestId("result-0-smiles").textContent).toBe(ASPIRIN.smiles);
  });

  it("searches ChEMBL when it is chosen", async () => {
    api.searchMolecules.mockResolvedValue(response([{ ...ASPIRIN, source: "chembl", id: "CHEMBL25" }], "chembl"));
    renderSearch();
    fireEvent.change(screen.getByTestId("search-source"), { target: { value: "chembl" } });
    search("aspirin");

    await screen.findByTestId("result-0");
    expect(api.searchMolecules).toHaveBeenCalledWith("aspirin", "chembl");
    expect(screen.getByTestId("result-0-id").textContent).toContain("CHEMBL25");
  });

  it("sends the chosen structure to the Submit page", async () => {
    api.searchMolecules.mockResolvedValue(response([ASPIRIN]));
    renderSearch();
    search("aspirin");
    await screen.findByTestId("result-0");
    fireEvent.click(screen.getByTestId("result-0-use"));

    expect(screen.getByTestId("submit-stub").textContent).toBe(ASPIRIN.smiles);
  });

  it("disables 'use this molecule' for a hit without a structure", async () => {
    api.searchMolecules.mockResolvedValue(response([NO_STRUCTURE]));
    renderSearch();
    search("aspirin");
    await screen.findByTestId("result-0");

    expect(screen.getByTestId("result-0-use").disabled).toBe(true);
    expect(screen.getByTestId("result-0-smiles").textContent).toBe("No structure available");
    expect(screen.getByTestId("result-0-name").textContent).toBe("Unnamed compound");
  });

  it("names the failing service and points to direct SMILES entry when it is down (504)", async () => {
    api.searchMolecules.mockRejectedValue(
      new ApiError({ status: 504, code: "upstream_timeout", message: "PubChem did not respond in time.", requestId: "req-9" }),
    );
    renderSearch();
    search("aspirin");

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("PubChem");
    expect(alert.textContent).toContain("req-9");
    expect(screen.getByTestId("search-fallback-link").getAttribute("href")).toBe("/submit");
    expect(screen.queryByTestId("search-results")).toBeNull();
  });

  it("shows a 400 without the 'service is down' hint", async () => {
    api.searchMolecules.mockRejectedValue(
      new ApiError({ status: 400, code: "validation_error", message: "The query must be at most 200 characters." }),
    );
    renderSearch();
    search("x");

    expect((await screen.findByRole("alert")).textContent).toContain("at most 200");
    expect(screen.queryByTestId("search-fallback")).toBeNull();
  });

  it("clears the previous results when a new search fails", async () => {
    api.searchMolecules
      .mockResolvedValueOnce(response([ASPIRIN]))
      .mockRejectedValueOnce(new ApiError({ status: 502, code: "upstream_error", message: "ChEMBL returned an error." }));
    renderSearch();
    search("aspirin");
    await screen.findByTestId("result-0");

    search("caffeine");
    await screen.findByRole("alert");
    expect(screen.queryByTestId("search-results")).toBeNull();
  });
});
