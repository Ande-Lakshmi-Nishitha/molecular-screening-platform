"""Unit tests for the real quantum backend and its result (MSP-28; FR-023..027, QA-ACC, TBD-C5, TBD-C6).

Two groups. The first needs no quantum libraries: the convergence rule, the labelled payload built from a
backend that is not the stub, the flag that keeps a non-converged run out of triage, and cancellation in the
worker. The second needs PennyLane and PySCF (requirements-qml.txt, Linux only) and is skipped without them:
H2 and LiH against full CI (QA-ACC: within 1.6 mHa), the classical references, and the not-converged path.

Run from the repo root with:  python -m unittest discover -s tests -t .
"""
# pylint: disable=missing-function-docstring,missing-class-docstring,import-outside-toplevel
import importlib.util
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.core import labels
from backend.registry.loader import ModelLoadError
from backend.workers.qml import predictor
from backend.workers.qml.backend import GapEstimate, QuantumBackend, QuantumCancelled
from backend.workers.qml.convergence import ConvergenceRecord
from backend.workers.qml.stub_backend import StubBackend

SETTINGS = SimpleNamespace(qml_conv_threshold_mha=1.6)
ETHANOL = "CCO"
HAVE_QUANTUM = all(importlib.util.find_spec(name) for name in ("pennylane", "pyscf"))

DETAILS = {
    "references": {"hf_energy_ha": -154.0, "hf_gap_ev": 25.0, "fci_ground_energy_ha": -154.1,
                   "fci_excited_energy_ha": -153.6, "fci_gap_ev": 13.98, "source": "PySCF test"},
    "run": {"mapping": "jordan_wigner", "ansatz": "UCCSD", "optimiser": "L-BFGS-B", "qubits": 8},
    "ground_state": {"energy_ha": -154.0999, "error_vs_fci_mha": 0.1, "iterations": 5,
                     "iteration_energies_ha": [-154.0, -154.0999], "iteration_errors_vs_fci_mha": [100.0, 0.1]},
    "excited_state": {"energy_ha": -153.6, "error_vs_fci_mha": 0.0},
    "gap_error_ev": 0.003,
}


class FakeVqe(QuantumBackend):
    """A backend that is not the stub, with a chosen error against full CI."""

    name, version, is_stub = "vqe", "9.9.9", False

    def __init__(self, error_mha=0.2):
        self.error_mha = error_mha
        self.seen_cancel = None

    def estimate_gap(self, smiles, threshold_mha, should_cancel=None):
        self.seen_cancel = should_cancel
        return GapEstimate(13.98, ConvergenceRecord(5, 0.01, threshold_mha, self.error_mha), 4, 4, DETAILS)


class ConvergenceTests(unittest.TestCase):
    def test_fr025_with_a_full_ci_reference_the_error_decides_not_the_last_change(self):
        self.assertTrue(ConvergenceRecord(5, 50.0, 1.6, 1.6).converged)
        self.assertFalse(ConvergenceRecord(5, 0.0, 1.6, 1.61).converged)

    def test_without_a_reference_the_last_change_decides(self):
        self.assertTrue(ConvergenceRecord(5, 1.6, 1.6).converged)
        self.assertFalse(ConvergenceRecord(5, 1.7, 1.6).converged)

    def test_the_error_is_in_the_block_only_when_there_is_one(self):
        self.assertNotIn("error_vs_fci_mha", ConvergenceRecord(5, 0.5, 1.6).to_dict())
        self.assertEqual(ConvergenceRecord(5, 0.5, 1.6, 0.2).to_dict(),
                         {"converged": True, "iterations": 5, "final_change_mha": 0.5, "threshold_mha": 1.6,
                          "error_vs_fci_mha": 0.2})


class RealPayloadTests(unittest.TestCase):
    def payload(self, error_mha=0.2):
        return predictor.QmlPredictor(FakeVqe(error_mha), 1.6).predict_smiles(ETHANOL)

    def test_fr026_the_payload_carries_the_value_the_references_and_the_run_settings(self):
        result = self.payload()
        payload = result.payload
        self.assertEqual(result.model_version, "qml_vqe@9.9.9")
        self.assertIs(payload["is_stub"], False)
        stability = payload["electronic_stability"]
        self.assertEqual((stability["metric"], stability["value"], stability["unit"]),
                         ("vqe_excitation_gap", 13.98, "eV"))
        self.assertEqual(stability["limitation"], labels.LIMITATION_QML)
        self.assertEqual(payload["classical_references"]["hf_gap_ev"], 25.0)
        self.assertEqual(payload["classical_references"]["fci_gap_ev"], 13.98)
        self.assertEqual(payload["classical_references"]["limitation"], labels.LIMITATION_QML_REFERENCE)
        self.assertEqual(payload["run"]["mapping"], "jordan_wigner")
        self.assertEqual(payload["energies"]["ground_state"]["iteration_energies_ha"], [-154.0, -154.0999])
        self.assertEqual(payload["energies"]["ground_state"]["iteration_errors_vs_fci_mha"], [100.0, 0.1])
        self.assertEqual(payload["active_space"], {"electrons": 4, "orbitals": 4})
        self.assertEqual(payload["backend"], {"name": "vqe", "version": "9.9.9", "is_stub": False})

    def test_fr027_a_converged_run_is_not_flagged_and_stays_in_triage(self):
        payload = self.payload(1.6).payload
        self.assertTrue(payload["convergence"]["converged"])
        self.assertIs(payload["excluded_from_triage"], False)
        self.assertNotIn("warning", payload["electronic_stability"])

    def test_fr027_a_run_above_the_threshold_is_flagged_and_excluded_from_triage(self):
        payload = self.payload(1.7).payload
        self.assertFalse(payload["convergence"]["converged"])
        self.assertEqual(payload["convergence"]["error_vs_fci_mha"], 1.7)
        self.assertIs(payload["excluded_from_triage"], True)
        self.assertEqual(payload["electronic_stability"]["warning"], labels.WARNING_QML_NOT_CONVERGED)
        self.assertEqual(payload["electronic_stability"]["value"], 13.98)   # the value is still returned

    def test_nfr007_nfr008_every_block_with_a_number_has_unit_limitation_and_model_version(self):
        payload = self.payload().payload
        for block in (payload["electronic_stability"], payload["classical_references"]):
            self.assertEqual(labels.missing_labels(block), [])

    def test_nfr011_the_real_wording_makes_no_claim_about_safety_or_effect(self):
        for text in (labels.LIMITATION_QML, labels.LIMITATION_QML_REFERENCE, labels.WARNING_QML_NOT_CONVERGED):
            self.assertEqual(labels.forbidden_claims_in(text), [])
            self.assertFalse(text.startswith("STUB"))

    def test_the_payload_is_json_safe(self):
        payload = self.payload().payload
        self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_the_params_hash_depends_on_the_backend_and_the_threshold(self):
        first = predictor.QmlPredictor(FakeVqe(), 1.6)
        self.assertEqual(first.params_hash, predictor.QmlPredictor(FakeVqe(), 1.6).params_hash)
        self.assertNotEqual(first.params_hash, predictor.QmlPredictor(FakeVqe(), 0.8).params_hash)

    def test_fr021_the_eligibility_rule_still_applies_to_a_real_backend(self):
        with self.assertRaises(ValueError):
            predictor.QmlPredictor(FakeVqe(), 1.6).predict_smiles("C" * 10)
        with self.assertRaises(ValueError):
            predictor.QmlPredictor(FakeVqe(), 1.6).predict_smiles("CCS")

    def test_the_cancel_check_reaches_the_backend(self):
        backend = FakeVqe()
        check = lambda: False  # noqa: E731
        predictor.QmlPredictor(backend, 1.6).predict_smiles(ETHANOL, check)
        self.assertIs(backend.seen_cancel, check)

    def test_a_not_converged_stub_keeps_its_agreed_fields_and_gets_no_warning(self):
        backend = StubBackend(final_change_mha=5.0)
        payload = predictor.QmlPredictor(backend, 1.6).predict_smiles(ETHANOL).payload
        self.assertFalse(payload["convergence"]["converged"])
        self.assertNotIn("warning", payload["electronic_stability"])
        self.assertNotIn("error_vs_fci_mha", payload["convergence"])

    def test_the_stub_payload_has_none_of_the_real_blocks(self):
        payload = predictor.get_predictor(settings=SETTINGS).predict_smiles(ETHANOL).payload
        for name in ("excluded_from_triage", "classical_references", "run", "energies"):
            self.assertNotIn(name, payload)
        self.assertEqual(payload["electronic_stability"]["metric"], "homo_lumo_gap")

    def test_the_vqe_backend_is_chosen_by_name(self):
        if HAVE_QUANTUM:
            self.assertEqual(predictor.get_predictor(backend_name="vqe", settings=SETTINGS).backend.name, "vqe")
        else:
            with self.assertRaises(ModelLoadError):
                predictor.get_predictor(backend_name="vqe", settings=SETTINGS)

    def test_without_the_quantum_libraries_the_backend_is_unavailable_with_no_fallback(self):
        with patch.dict(sys.modules, {"backend.workers.qml.vqe_backend": None}):
            with self.assertRaises(ModelLoadError):
                predictor.get_predictor(backend_name="vqe", settings=SETTINGS)


@unittest.skipUnless(HAVE_QUANTUM, "needs PennyLane and PySCF (requirements-qml.txt, Linux only)")
class AccuracyTests(unittest.TestCase):
    """QA-ACC: H2 and LiH within 1.6 mHa of full CI. LiH takes about 20 s."""

    @classmethod
    def setUpClass(cls):
        from backend.workers.qml.vqe_backend import VqeBackend
        backend = VqeBackend()
        cls.h2 = backend.compute(["H", "H"], [[0, 0, 0], [0, 0, 0.74]], 1.6)
        cls.lih = backend.compute(["Li", "H"], [[0, 0, 0], [0, 0, 1.595]], 1.6)

    def test_qa_acc_h2_energies_are_within_the_threshold_of_full_ci(self):
        details = self.h2.details
        self.assertLessEqual(abs(details["ground_state"]["error_vs_fci_mha"]), 1.6)
        self.assertLessEqual(abs(details["excited_state"]["error_vs_fci_mha"]), 1.6)
        self.assertTrue(self.h2.convergence.converged)
        self.assertEqual((self.h2.active_electrons, self.h2.active_orbitals), (2, 2))

    def test_qa_acc_lih_energies_are_within_the_threshold_of_full_ci(self):
        details = self.lih.details
        self.assertLessEqual(abs(details["ground_state"]["error_vs_fci_mha"]), 1.6)
        self.assertLessEqual(abs(details["excited_state"]["error_vs_fci_mha"]), 1.6)
        self.assertTrue(self.lih.convergence.converged)

    def test_lih_uses_the_4_electron_4_orbital_active_space_on_8_qubits(self):
        self.assertEqual((self.lih.active_electrons, self.lih.active_orbitals), (4, 4))
        self.assertEqual(self.lih.details["run"]["qubits"], 8)

    def test_the_gap_is_close_to_the_full_ci_gap(self):
        for estimate in (self.h2, self.lih):
            reference = estimate.details["references"]["fci_gap_ev"]
            self.assertAlmostEqual(estimate.gap_ev, reference, delta=0.01)
            self.assertLess(abs(estimate.details["gap_error_ev"]), 0.01)

    def test_the_hartree_fock_and_full_ci_gaps_are_stored_as_references(self):
        for estimate in (self.h2, self.lih):
            references = estimate.details["references"]
            for key in ("hf_gap_ev", "fci_gap_ev", "hf_energy_ha", "fci_ground_energy_ha"):
                self.assertIn(key, references)
        self.assertGreater(self.lih.details["references"]["hf_gap_ev"], self.lih.details["references"]["fci_gap_ev"])

    def test_the_run_settings_and_iteration_energies_are_stored(self):
        run = self.lih.details["run"]
        self.assertEqual((run["mapping"], run["basis"]), ("jordan_wigner", "sto-3g"))
        self.assertIn("UCCSD", run["ansatz"])
        self.assertEqual(run["optimiser"], "L-BFGS-B")
        self.assertIn("statevector", run["simulator"])
        ground = self.lih.details["ground_state"]
        self.assertEqual(len(ground["iteration_energies_ha"]), len(ground["iteration_errors_vs_fci_mha"]))
        self.assertGreater(len(ground["iteration_energies_ha"]), 1)
        self.assertGreater(ground["iteration_energies_ha"][0], ground["iteration_energies_ha"][-1])


@unittest.skipUnless(HAVE_QUANTUM, "needs PennyLane and PySCF (requirements-qml.txt, Linux only)")
class BackendBehaviourTests(unittest.TestCase):
    def setUp(self):
        from backend.workers.qml.vqe_backend import VqeBackend
        self.backend = VqeBackend()

    def test_fr027_an_optimiser_stopped_at_the_start_gives_not_converged(self):
        from scipy.optimize import OptimizeResult
        from backend.workers.qml import vqe_backend

        def stay_at_the_start(fun, x0, **_kwargs):   # the optimiser never moves: the result is the Hartree-Fock energy
            return OptimizeResult(x=x0, fun=fun(x0)[0], nit=0)

        with patch.object(vqe_backend, "minimize", stay_at_the_start):
            estimate = self.backend.compute(["H", "H"], [[0, 0, 0], [0, 0, 0.74]], 1.6)
        self.assertFalse(estimate.convergence.converged)
        self.assertGreater(estimate.convergence.error_vs_fci_mha, 1.6)
        self.assertGreater(estimate.gap_ev, 0)   # the value is still returned

    def test_a_cancelled_run_stops_with_quantum_cancelled(self):
        with self.assertRaises(QuantumCancelled):
            self.backend.compute(["H", "H"], [[0, 0, 0], [0, 0, 0.74]], 1.6, should_cancel=lambda: True)

    def test_an_open_shell_molecule_is_refused(self):
        with self.assertRaises(ValueError):
            self.backend.estimate_gap("[CH3]", 1.6)

    def test_an_empty_or_unreadable_structure_is_refused(self):
        for bad in ("", "  ", "not a smiles"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.backend.estimate_gap(bad, 1.6)

    def test_the_same_structure_gives_the_same_geometry(self):
        from backend.workers.qml.vqe_backend import geometry
        self.assertEqual(geometry(ETHANOL), geometry(ETHANOL))

    def test_the_identity_names_the_libraries_and_the_method(self):
        identity = self.backend.identity()
        for key in ("pennylane", "pyscf", "ansatz", "optimiser", "mapping", "active_electrons", "excited_method"):
            self.assertIn(key, identity)
        self.assertFalse(identity["is_stub"])

    def test_the_active_space_never_exceeds_the_molecule(self):
        self.assertEqual(self.backend._active_space(["H", "H"], 0), (2, 2))      # pylint: disable=protected-access
        self.assertEqual(self.backend._active_space(["Li", "H"], 0), (4, 4))     # pylint: disable=protected-access
        self.assertEqual(self.backend._active_space(list("CCO") + ["H"] * 6, 0), (4, 4))  # pylint: disable=protected-access

    def test_a_molecule_through_the_predictor_gives_a_complete_labelled_payload(self):
        result = predictor.get_predictor(backend_name="vqe", settings=SETTINGS).predict_smiles("CO")
        payload = result.payload
        self.assertEqual(result.model_version, "qml_vqe@1.0.0")
        self.assertEqual(payload["convergence"]["converged"], not payload["excluded_from_triage"])
        self.assertEqual(labels.missing_labels(payload["electronic_stability"]), [])
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertEqual(payload["active_space"], {"electrons": 4, "orbitals": 4})


if __name__ == "__main__":
    unittest.main()
