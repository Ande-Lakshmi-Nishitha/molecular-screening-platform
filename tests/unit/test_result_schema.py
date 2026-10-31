"""Schema test over every tier: each result block is labelled and no text makes a forbidden claim (MSP-32).

Covers NFR-007, NFR-008, NFR-009, NFR-011, BR-006. One real payload per tier is produced with the
tiny real models from the test fixtures (ML, DL) and the stub backend (QML), then walked:

  * every block that has a unit, a limitation or a model_version has all three, non-empty
  * the labelled blocks each tier must have are present
  * no text anywhere in a payload, and no sentence in core/labels.py, uses a forbidden claim word
  * the job status body carries the disclaimer

The DL part needs torch and is skipped without it.

Run from the repo root with:  python -m unittest discover -s tests -t .
"""
# pylint: disable=missing-function-docstring,missing-class-docstring
import json
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.api.jobs import status
from backend.core import labels
from backend.db.models import Base, Job, JobTask, Molecule, Result, User
from backend.workers.ml import predictor as ml_predictor
from backend.workers.qml import predictor as qml_predictor
from tests.unit.training.synthetic import make_frame
from tests.unit.workers_ml import ml_fixtures as ml_fx

try:
    import torch  # noqa: F401  # pylint: disable=unused-import
    from backend.workers.dl import predictor as dl_predictor
    from tests.unit.workers_dl import dl_fixtures as dl_fx
except ImportError:  # CI installs requirements.txt only
    dl_predictor = dl_fx = None

LABEL_KEYS = ("unit", "limitation", "model_version")
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ETHANOL = "CCO"


def blocks(value, path=""):
    """Every dict in a payload that holds at least one label key, with its path."""
    found = []
    if isinstance(value, dict):
        if any(key in value for key in LABEL_KEYS):
            found.append((path or "(top)", value))
        for key, item in value.items():
            found += blocks(item, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found += blocks(item, f"{path}[{index}]")
    return found


def texts(value):
    """Every string in a payload (keys included)."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from texts(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from texts(item)
    elif isinstance(value, str):
        yield value


def make_payloads():
    """One real payload per tier, as {tier: (payload, [paths of the blocks that must be labelled])}."""
    root = ml_fx.built_root()
    ml_model = ml_predictor.MlPredictor.load(ml_fx.models_dir(root), cards_dir=root / "cards")
    out = {"ml": (ml_model.predict_smiles(make_frame()["smiles_std"].tolist()[0]).payload,
                  ["solubility", "tox21", "alerts", "rules", "domain"]),
           "qml": (qml_predictor.get_predictor().predict_smiles(ETHANOL).payload, ["electronic_stability"])}
    if dl_predictor is not None:
        dl_root = dl_fx.built_root()
        model = dl_predictor.get_predictor("TEST_TARGET", dl_fx.models_dir(dl_root), dl_root / "cards", dl_fx.TARGETS)
        out["dl"] = (model.predict_smiles(ASPIRIN).payload, ["activity", "baseline"])
    return out


class SchemaCase(unittest.TestCase):
    payloads = {}

    @classmethod
    def setUpClass(cls):
        cls.payloads = make_payloads()


class PayloadSchemaTests(SchemaCase):
    def test_each_payload_names_its_tier_and_is_json_safe(self):
        for tier, (payload, _) in self.payloads.items():
            with self.subTest(tier=tier):
                self.assertEqual(payload["tier"], tier)
                self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_nfr007_nfr008_every_labelled_block_has_unit_limitation_and_model_version(self):
        for tier, (payload, _) in self.payloads.items():
            for path, block in blocks(payload):
                with self.subTest(tier=tier, block=path):
                    self.assertEqual(labels.missing_labels(block), [])
                    for key in LABEL_KEYS:
                        self.assertIsInstance(block[key], str)

    def test_the_blocks_each_tier_must_have_are_present_and_labelled(self):
        for tier, (payload, required) in self.payloads.items():
            for name in required:
                with self.subTest(tier=tier, block=name):
                    self.assertIn(name, payload)
                    self.assertEqual(labels.missing_labels(payload[name]), [])

    def test_br006_tox21_values_carry_the_assay_signal_label(self):
        self.assertEqual(self.payloads["ml"][0]["tox21"]["label"], labels.ASSAY_SIGNAL)

    def test_the_dl_unit_is_the_targets_own_and_the_baseline_uses_the_same(self):
        if dl_predictor is None:
            self.skipTest("torch is not installed")
        payload = self.payloads["dl"][0]
        self.assertIn(payload["activity"]["unit"], ("pIC50", "pKi"))
        self.assertEqual(payload["baseline"]["unit"], payload["activity"]["unit"])

    def test_nfr011_no_text_in_any_payload_makes_a_forbidden_claim(self):
        for tier, (payload, _) in self.payloads.items():
            with self.subTest(tier=tier):
                found = [(text, labels.forbidden_claims_in(text)) for text in texts(payload)
                         if labels.forbidden_claims_in(text)]
                self.assertEqual(found, [])

    def test_the_qml_stub_is_marked_in_every_place_and_the_real_wording_is_not_the_stub_wording(self):
        payload = self.payloads["qml"][0]
        self.assertTrue(payload["is_stub"] and payload["backend"]["is_stub"])
        self.assertTrue(payload["electronic_stability"]["limitation"].startswith("STUB:"))
        self.assertFalse(labels.LIMITATION_QML.startswith("STUB:"))


class WordingTests(unittest.TestCase):
    def test_all_wording_lists_every_fixed_sentence(self):
        wording = labels.all_wording()
        for name in ("DISCLAIMER", "LIMITATION_SOLUBILITY", "LIMITATION_TOX21", "LIMITATION_ACTIVITY",
                     "LIMITATION_BASELINE", "LIMITATION_QML", "LIMITATION_QML_STUB", "LIMITATION_ALERTS",
                     "LIMITATION_RULES", "LIMITATION_DOMAIN", "WARNING_OUT_OF_DOMAIN",
                     "WARNING_DOMAIN_UNAVAILABLE"):
            self.assertIn(name, wording)
        for name, text in wording.items():
            with self.subTest(name=name):
                self.assertGreater(len(text), 40)
                self.assertTrue(text.rstrip().endswith("."))
                self.assertEqual(labels.forbidden_claims_in(text), [])

    def test_nothing_that_is_not_a_sentence_is_listed(self):
        self.assertNotIn("FORBIDDEN_CLAIMS", labels.all_wording())
        self.assertNotIn("UNIT_EV", labels.all_wording())

    def test_the_forbidden_word_finder_matches_whole_words_only(self):
        self.assertEqual(labels.forbidden_claims_in("This molecule is SAFE and effective."), ["safe", "effective"])
        self.assertEqual(labels.forbidden_claims_in("A safe-looking, non-toxic compound"), ["safe", "non-toxic"])
        for clean in ("secure", "unsafe handling", "ineffective", "an approval step", "accurate", "safety review", "curedx", ""):
            with self.subTest(text=clean):
                self.assertEqual(labels.forbidden_claims_in(clean), [])

    def test_the_disclaimer_says_the_values_are_estimates_and_not_experimental_results(self):
        self.assertIn("preliminary computational estimates", labels.DISCLAIMER)
        self.assertIn("not experimental results", labels.DISCLAIMER)

    def test_qml_labels_pick_the_stub_or_the_real_wording(self):
        stub = labels.qml_labels(True, "qml_stub@1.0.0")
        real = labels.qml_labels(False, "qml_vqe@1.0.0")
        self.assertEqual((stub["limitation"], real["limitation"]), (labels.LIMITATION_QML_STUB, labels.LIMITATION_QML))
        for block in (stub, real):
            self.assertEqual(labels.missing_labels(block), [])
            self.assertEqual(block["unit"], "eV")


class StatusDisclaimerTests(unittest.TestCase):
    def test_nfr011_the_job_status_body_carries_the_disclaimer(self):
        engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            owner = User(email="s@example.com", password_hash="x")
            molecule = Molecule(inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N", canonical_smiles="CCO")
            session.add_all([owner, molecule])
            session.flush()
            job = Job(molecule_id=molecule.id, user_id=owner.id, options={}, rules_version="0.1")
            session.add(job)
            session.flush()
            task = JobTask(job_id=job.id, tier="ml", status="done")
            session.add(task)
            session.flush()
            session.add(Result(job_task_id=task.id, model_version="1.0", params_hash="abc", payload={"x": 1}))
            session.commit()
            body = status.build_status(session, job.id, owner.id)
        self.assertEqual(body["disclaimer"], labels.DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
