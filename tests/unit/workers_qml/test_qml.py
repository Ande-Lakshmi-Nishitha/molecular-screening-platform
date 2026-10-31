"""Unit tests for the QML tier: backend interface, stub, convergence, payload contract (MSP-27).

Covers FR-021, FR-023, FR-025, FR-027, BR-006, NFR-007, NFR-008, NFR-011. The payload test pins the
exact field names agreed with the aggregator (MSP-31): change them only together with Ayushi.

Run from the repo root with:  python -m unittest discover -s tests -t .
"""
# pylint: disable=missing-function-docstring,missing-class-docstring
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.core import labels
from backend.registry.loader import ModelLoadError
from backend.workers.qml import predictor
from backend.workers.qml.backend import GapEstimate, QuantumBackend, QuantumCancelled
from backend.workers.qml.convergence import ConvergenceRecord
from backend.workers.qml.stub_backend import StubBackend

ETHANOL = "CCO"
SETTINGS = SimpleNamespace(qml_conv_threshold_mha=1.6)


class ConvergenceTests(unittest.TestCase):
    def test_fr025_converged_when_the_last_change_is_at_or_below_the_threshold(self):
        self.assertTrue(ConvergenceRecord(10, 1.0, 1.6).converged)
        self.assertTrue(ConvergenceRecord(10, 1.6, 1.6).converged)
        self.assertFalse(ConvergenceRecord(10, 1.61, 1.6).converged)

    def test_the_block_has_the_agreed_fields(self):
        self.assertEqual(ConvergenceRecord(12, 0.9, 1.6).to_dict(),
                         {"converged": True, "iterations": 12, "final_change_mha": 0.9, "threshold_mha": 1.6})


class StubBackendTests(unittest.TestCase):
    def test_it_returns_the_same_fixed_values_for_every_molecule(self):
        first = StubBackend().estimate_gap("CCO", 1.6)
        second = StubBackend().estimate_gap("CC(=O)N", 1.6)
        self.assertEqual(first, second)
        self.assertEqual((first.gap_ev, first.convergence.iterations), (4.5, 12))
        self.assertEqual((first.active_electrons, first.active_orbitals), (4, 4))

    def test_it_identifies_itself_as_a_stub(self):
        backend = StubBackend()
        self.assertTrue(backend.is_stub)
        self.assertEqual((backend.name, backend.version), ("stub", "1.0.0"))
        self.assertTrue(backend.identity()["is_stub"])

    def test_a_larger_final_change_gives_not_converged(self):
        estimate = StubBackend(final_change_mha=2.5).estimate_gap("CCO", 1.6)
        self.assertFalse(estimate.convergence.converged)
        self.assertEqual(estimate.gap_ev, 4.5)

    def test_it_checks_for_cancellation_in_every_iteration(self):
        calls = []
        StubBackend().estimate_gap("CCO", 1.6, lambda: calls.append(1) or False)
        self.assertEqual(len(calls), 12)
        with self.assertRaises(QuantumCancelled):
            StubBackend().estimate_gap("CCO", 1.6, lambda: True)

    def test_an_empty_structure_is_refused(self):
        for bad in ("", "  ", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                StubBackend().estimate_gap(bad, 1.6)

    def test_the_interface_cannot_be_used_without_estimate_gap(self):
        with self.assertRaises(TypeError):
            QuantumBackend()  # pylint: disable=abstract-class-instantiated
        self.assertEqual(GapEstimate(1.0, ConvergenceRecord(1, 0.1, 1.6), 2, 2).gap_ev, 1.0)


class PayloadContractTests(unittest.TestCase):
    """The names below are what the aggregator reads (MSP-31)."""

    def setUp(self):
        self.result = predictor.get_predictor(settings=SETTINGS).predict_smiles(ETHANOL)
        self.payload = self.result.payload

    def test_the_payload_has_exactly_the_agreed_fields(self):
        self.assertEqual(set(self.payload),
                         {"tier", "is_stub", "electronic_stability", "convergence", "backend", "active_space"})
        self.assertEqual(set(self.payload["electronic_stability"]),
                         {"metric", "value", "unit", "limitation", "model_version"})
        self.assertEqual(set(self.payload["convergence"]),
                         {"converged", "iterations", "final_change_mha", "threshold_mha"})
        self.assertEqual(set(self.payload["backend"]), {"name", "version", "is_stub"})
        self.assertEqual(set(self.payload["active_space"]), {"electrons", "orbitals"})

    def test_the_values_of_the_stub(self):
        self.assertEqual(self.payload["tier"], "qml")
        stability = self.payload["electronic_stability"]
        self.assertEqual((stability["metric"], stability["value"], stability["unit"]), ("homo_lumo_gap", 4.5, "eV"))
        self.assertEqual(self.payload["convergence"],
                         {"converged": True, "iterations": 12, "final_change_mha": 0.9, "threshold_mha": 1.6})
        self.assertEqual(self.payload["active_space"], {"electrons": 4, "orbitals": 4})

    def test_the_stub_is_marked_in_every_place_the_aggregator_can_look(self):
        self.assertIs(self.payload["is_stub"], True)
        self.assertIs(self.payload["backend"]["is_stub"], True)
        self.assertTrue(self.payload["electronic_stability"]["limitation"].startswith("STUB:"))
        self.assertEqual(self.result.model_version, "qml_stub@1.0.0")
        self.assertEqual(self.payload["electronic_stability"]["model_version"], "qml_stub@1.0.0")

    def test_nfr007_nfr008_the_block_carries_unit_limitation_and_model_version(self):
        self.assertEqual(labels.missing_labels(self.payload["electronic_stability"]), [])

    def test_the_payload_is_json_safe(self):
        self.assertEqual(json.loads(json.dumps(self.payload)), self.payload)

    def test_the_threshold_comes_from_the_settings(self):
        result = predictor.get_predictor(settings=SimpleNamespace(qml_conv_threshold_mha=0.5)).predict_smiles(ETHANOL)
        self.assertEqual(result.payload["convergence"]["threshold_mha"], 0.5)
        self.assertFalse(result.payload["convergence"]["converged"])   # 0.9 mHa > 0.5 mHa

    def test_the_params_hash_is_stable_and_depends_on_the_settings(self):
        again = predictor.get_predictor(settings=SETTINGS).predict_smiles("CC(=O)N")
        self.assertEqual(self.result.params_hash, again.params_hash)
        self.assertEqual(len(self.result.params_hash), 64)
        other = predictor.get_predictor(settings=SimpleNamespace(qml_conv_threshold_mha=0.5))
        self.assertNotEqual(self.result.params_hash, other.params_hash)

    def test_the_model_version_fits_the_database_column(self):
        self.assertLessEqual(len(self.result.model_version), 50)


class EligibilityAndLoadingTests(unittest.TestCase):
    def setUp(self):
        self.model = predictor.get_predictor(settings=SETTINGS)

    def test_fr021_a_molecule_that_does_not_qualify_is_refused(self):
        for smiles in ("CCCCCCCCCC", "CCS", "c1ccccc1Cl"):   # 10 heavy atoms, sulfur, chlorine
            with self.subTest(smiles=smiles), self.assertRaises(ValueError):
                self.model.predict_smiles(smiles)

    def test_a_molecule_at_the_limit_is_accepted(self):
        self.assertEqual(self.model.predict_smiles("CCCCCCCCC").payload["tier"], "qml")   # 9 carbons

    def test_an_unreadable_or_empty_structure_is_refused(self):
        for bad in ("not a smiles", "", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.model.predict_smiles(bad)

    def test_an_unknown_backend_is_refused_without_a_fallback(self):
        with self.assertRaises(ModelLoadError):
            predictor.get_predictor(backend_name="vqe", settings=SETTINGS)
        with patch.dict(os.environ, {"QML_BACKEND": "something-else"}):
            with self.assertRaises(ModelLoadError):
                predictor.get_predictor(settings=SETTINGS)

    def test_the_backend_is_chosen_by_name_or_environment(self):
        self.assertEqual(predictor.get_predictor(backend_name=" STUB ", settings=SETTINGS).backend.name, "stub")
        with patch.dict(os.environ, {"QML_BACKEND": "stub"}):
            self.assertEqual(predictor.get_predictor(settings=SETTINGS).backend.name, "stub")

    def test_the_default_threshold_is_read_from_the_real_settings(self):
        self.assertEqual(predictor.get_predictor().threshold_mha, 1.6)

    def test_a_non_stub_backend_has_no_wording_yet(self):
        class Real(StubBackend):
            is_stub = False
        with self.assertRaises(ModelLoadError):
            predictor.QmlPredictor(Real(), 1.6).predict_smiles(ETHANOL)


class WordingTests(unittest.TestCase):
    def test_nfr011_the_stub_wording_makes_no_claim_about_safety_or_effect(self):
        for word in ("safe", "non-toxic", "harmless", "effective", "approved", "cure"):
            self.assertNotIn(word, labels.LIMITATION_QML_STUB.lower())

    def test_the_stub_wording_says_it_is_not_a_calculation_and_not_for_ranking(self):
        text = labels.LIMITATION_QML_STUB
        self.assertTrue(text.startswith("STUB:"))
        self.assertIn("No quantum calculation was run", text)
        self.assertIn("must not be used to compare or rank", text)
        self.assertEqual(labels.qml_stub_labels("qml_stub@1.0.0")["unit"], "eV")


if __name__ == "__main__":
    unittest.main()
