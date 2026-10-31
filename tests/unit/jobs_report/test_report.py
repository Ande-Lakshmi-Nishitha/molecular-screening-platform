"""Unit tests for the job report builder and PDF renderer (MSP-38; FR-039). No database needed."""
import importlib.util
import json
import unittest
from datetime import datetime, timezone

from backend.api.jobs.report import FORMATS, REPORT_VERSION, _flatten, assemble_report, render_pdf
from backend.core.labels import DISCLAIMER

STATUS = {
    "job_id": "7f6c1f0e-0000-4000-8000-000000000001",
    "status": "partial",
    "target_id": "EGFR",
    "rules_version": "1.2",
    "created_at": "2026-11-02T10:00:00+00:00",
    "finished_at": "2026-11-02T10:01:00+00:00",
    "cancelled_at": None,
    "summary": None,
    "tasks": [
        {"id": "t1", "tier": "ml", "status": "done", "skip_reason": None, "error": None,
         "result": {"model_version": "1.0.0", "params_hash": "abc123",
                    "payload": {"solubility": {"value": -2.5, "unit": "log10(mol/L)", "limitation": "An estimate."}}}},
        {"id": "t2", "tier": "dl", "status": "skipped", "skip_reason": "NO_TARGET_SELECTED", "error": None, "result": None},
        {"id": "t3", "tier": "qml", "status": "failed", "skip_reason": None, "error": "TIMEOUT", "result": None},
    ],
}
MOLECULE = {"canonical_smiles": "CCO", "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N", "formula": "C2H6O", "heavy_atoms": 3}
NOW = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)


class AssembleReportTests(unittest.TestCase):
    def setUp(self):
        self.report = assemble_report(STATUS, MOLECULE, NOW)

    def test_carries_the_disclaimer_and_version_info(self):
        self.assertEqual(self.report["disclaimer"], DISCLAIMER)
        self.assertEqual(self.report["report_version"], REPORT_VERSION)
        self.assertEqual(self.report["versions"]["rules_version"], "1.2")
        self.assertEqual(self.report["versions"]["models"], {"ml": "1.0.0"})
        self.assertEqual(self.report["generated_at"], NOW.isoformat())

    def test_keeps_every_tier_including_skipped_and_failed_ones(self):
        by_tier = {tier["tier"]: tier for tier in self.report["tiers"]}
        self.assertEqual(list(by_tier), ["ml", "dl", "qml"])
        self.assertEqual(by_tier["ml"]["params_hash"], "abc123")
        self.assertEqual(by_tier["ml"]["result"]["solubility"]["value"], -2.5)
        self.assertEqual(by_tier["dl"]["skip_reason"], "NO_TARGET_SELECTED")
        self.assertEqual(by_tier["qml"]["error"], "TIMEOUT")
        self.assertIsNone(by_tier["qml"]["model_version"])

    def test_holds_the_molecule_and_job_fields(self):
        self.assertEqual(self.report["molecule"]["canonical_smiles"], "CCO")
        self.assertEqual(self.report["job"]["id"], STATUS["job_id"])
        self.assertEqual(self.report["job"]["status"], "partial")
        self.assertEqual(self.report["job"]["target_id"], "EGFR")

    def test_is_json_serialisable(self):
        self.assertEqual(json.loads(json.dumps(self.report))["job"]["id"], STATUS["job_id"])

    def test_formats(self):
        self.assertEqual(set(FORMATS), {"json", "pdf"})


class FlattenTests(unittest.TestCase):
    def test_nested_payload_becomes_dotted_rows(self):
        rows = dict(_flatten({"a": {"b": 1, "c": [1, 2]}, "d": None, "e": [{"x": 1}]}))
        self.assertEqual(rows, {"a.b": "1", "a.c": "1, 2", "d": "", "e[0].x": "1"})


@unittest.skipUnless(importlib.util.find_spec("reportlab"), "reportlab is not installed")
class RenderPdfTests(unittest.TestCase):
    def test_produces_a_pdf(self):
        pdf = render_pdf(assemble_report(STATUS, MOLECULE, NOW))
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(len(pdf), 1000)

    def test_handles_a_job_with_no_results(self):
        status = {**STATUS, "tasks": [{"id": "t1", "tier": "ml", "status": "queued", "result": None}]}
        self.assertTrue(render_pdf(assemble_report(status, MOLECULE, NOW)).startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
