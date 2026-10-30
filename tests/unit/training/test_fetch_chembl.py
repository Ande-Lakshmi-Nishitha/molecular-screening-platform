"""Tests for the ChEMBL download and cleaning script (MSP-25; BR-009). No internet is used."""
# pylint: disable=missing-function-docstring,missing-class-docstring,protected-access
import json
import tempfile
import unittest
from pathlib import Path

import requests

from training import fetch_chembl as fc
from training.utils import file_sha256


def record(smiles="CCO", pchembl="6.5", **overrides):
    base = {
        "molecule_chembl_id": "CHEMBL1", "canonical_smiles": smiles, "pchembl_value": pchembl,
        "standard_type": "IC50", "standard_relation": "=", "assay_type": "B",
        "data_validity_comment": None, "potential_duplicate": 0, "target_chembl_id": "CHEMBL203",
        "assay_chembl_id": "CHEMBL9",
    }
    base.update(overrides)
    return base


class FakeResponse:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status

    def json(self):
        return self._body


class FakeSession:
    """Returns the queued responses in order and remembers the requests."""

    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, timeout=None):  # pylint: disable=unused-argument
        self.calls.append((url, params))
        if url.endswith("status.json"):
            return FakeResponse({"chembl_db_version": "ChEMBL_TEST"})
        return self.responses.pop(0)


def page(items, following=None):
    return FakeResponse({"activities": items, "page_meta": {"next": following}})


class FetchTests(unittest.TestCase):
    def test_pages_are_followed_until_there_is_no_next_link(self):
        session = FakeSession([page([record("CCO")], "/chembl/api/data/activity.json?offset=1"), page([record("CCN")])])
        records = fc.fetch_records("CHEMBL203", "IC50", session, pause=0)
        self.assertEqual([r["canonical_smiles"] for r in records], ["CCO", "CCN"])
        self.assertEqual(session.calls[0][1]["target_chembl_id"], "CHEMBL203")
        self.assertEqual(session.calls[0][1]["standard_type"], "IC50")
        self.assertEqual(session.calls[0][1]["pchembl_value__isnull"], "false")
        self.assertEqual(session.calls[1][0], "https://www.ebi.ac.uk/chembl/api/data/activity.json?offset=1")
        self.assertIsNone(session.calls[1][1])

    def test_only_the_wanted_fields_are_kept(self):
        item = record()
        item["canonical_smiles_extra"] = "x"
        records = fc.fetch_records("CHEMBL203", "IC50", FakeSession([page([item])]), pause=0)
        self.assertEqual(set(records[0]), set(fc.KEPT_FIELDS))

    def test_http_error_stops_with_a_clear_message(self):
        with self.assertRaises(fc.DataError) as caught:
            fc.fetch_records("CHEMBL203", "IC50", FakeSession([FakeResponse({}, 503)]), pause=0)
        self.assertIn("503", str(caught.exception))

    def test_endless_paging_is_stopped(self):
        pages = [page([record()], "/next") for _ in range(5)]
        with self.assertRaises(fc.DataError):
            fc.fetch_records("CHEMBL203", "IC50", FakeSession(pages), pause=0, max_pages=3)

    def test_chembl_release_survives_a_failing_status_call(self):
        class Broken:
            def get(self, *_args, **_kwargs):
                raise requests.ConnectionError("down")

        self.assertIsNone(fc.chembl_release(Broken()))
        self.assertEqual(fc.chembl_release(FakeSession([])), "ChEMBL_TEST")

    def test_make_session_retries_and_identifies_itself(self):
        session = fc.make_session()
        self.assertIn("msp-training", session.headers["User-Agent"])
        self.assertEqual(session.get_adapter("https://x").max_retries.total, 5)


class CurateTests(unittest.TestCase):
    def test_filters_drop_what_chembl_marks_as_unreliable(self):
        records = [
            record("CCO"),
            record("CCN", standard_relation=">"),
            record("CCC", assay_type="F"),
            record("CCCC", data_validity_comment="Outside typical range"),
            record("CCCCC", potential_duplicate=1),
            record("CCCCCC", standard_type="Ki"),
            record("", pchembl="6.0"),
            record("CCCCCCC", pchembl=None),
        ]
        table, stats = fc.curate(records, "IC50")
        self.assertEqual(table["smiles_std"].tolist(), ["CCO"])
        self.assertEqual(stats["n_downloaded"], 8)
        self.assertEqual(stats["n_dropped_filters"], 7)

    def test_values_outside_the_plausible_range_are_dropped(self):
        table, stats = fc.curate([record("CCO", "6.0"), record("CCN", "1.0"), record("CCC", "13.5")], "IC50")
        self.assertEqual(len(table), 1)
        self.assertEqual(stats["n_dropped_out_of_range"], 2)

    def test_unparsable_and_unsupported_structures_are_counted(self):
        records = [record("CCO"), record("not a molecule"), record("C[Si](C)(C)C"), record("CCCl")]
        table, stats = fc.curate(records, "IC50")
        self.assertEqual(sorted(table["smiles_std"]), ["CCCl", "CCO"])
        self.assertEqual(stats["n_dropped_unparsable"], 1)
        self.assertEqual(stats["n_dropped_unsupported_atoms"], 1)

    def test_repeated_measurements_become_their_median(self):
        records = [record("CCO", "6.0"), record("OCC", "7.0"), record("C(C)O", "6.5")]
        table, stats = fc.curate(records, "IC50")
        self.assertEqual(len(table), 1)
        self.assertAlmostEqual(table["pactivity"][0], 6.5)
        self.assertEqual(table["n_measurements"][0], 3)
        self.assertEqual(stats["n_measurements_used"], 3)

    def test_molecules_whose_repeats_disagree_are_dropped(self):
        records = [record("CCO", "5.0"), record("OCC", "8.0"), record("CCN", "6.0")]
        table, stats = fc.curate(records, "IC50")
        self.assertEqual(table["smiles_std"].tolist(), ["CCN"])
        self.assertEqual(stats["n_dropped_inconsistent_repeats"], 1)

    def test_same_molecule_written_two_ways_is_one_row(self):
        table, _ = fc.curate([record("OC(=O)C"), record("CC(O)=O")], "IC50")
        self.assertEqual(len(table), 1)

    def test_empty_input_gives_an_empty_table(self):
        table, stats = fc.curate([], "IC50")
        self.assertEqual(len(table), 0)
        self.assertEqual(stats["n_clean"], 0)

    def test_pactivity_is_numeric(self):
        table, _ = fc.curate([record("CCO", "6.25")], "IC50")
        self.assertEqual(float(table["pactivity"][0]), 6.25)


class TargetListTests(unittest.TestCase):
    def write(self, text):
        folder = Path(tempfile.mkdtemp())
        path = folder / "targets.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_all_targets_or_a_selected_one(self):
        path = self.write("targets:\n  - {id: A, chembl_id: CHEMBL1, unit: pIC50}\n  - {id: B, chembl_id: CHEMBL2, unit: pKi}\n")
        self.assertEqual(len(fc.target_list(path)), 2)
        self.assertEqual(fc.target_list(path, "B")[0]["chembl_id"], "CHEMBL2")
        self.assertEqual(fc.target_list(path, "CHEMBL1")[0]["id"], "A")

    def test_unknown_target_is_an_error(self):
        path = self.write("targets:\n  - {id: A, chembl_id: CHEMBL1, unit: pIC50}\n")
        with self.assertRaises(fc.DataError):
            fc.target_list(path, "Z")

    def test_missing_field_or_bad_unit_is_an_error(self):
        with self.assertRaises(fc.DataError):
            fc.target_list(self.write("targets:\n  - {id: A, unit: pIC50}\n"))
        with self.assertRaises(fc.DataError):
            fc.target_list(self.write("targets:\n  - {id: A, chembl_id: CHEMBL1, unit: pEC50}\n"))

    def test_empty_file_means_no_targets(self):
        self.assertEqual(fc.target_list(self.write("targets: []\n")), [])


class RunTargetTests(unittest.TestCase):
    TARGET = {"id": "EGFR", "chembl_id": "CHEMBL203", "unit": "pIC50"}

    def items(self, count=30):
        smiles = ["C" * n + "O" for n in range(1, count + 1)]
        return [record(value, str(5.0 + (index % 40) / 10)) for index, value in enumerate(smiles)]

    def test_writes_raw_csv_and_meta_files(self):
        folder = Path(tempfile.mkdtemp())
        session = FakeSession([page(self.items())])
        meta = fc.run_target(self.TARGET, folder, session, min_rows=10)
        self.assertEqual(meta["n_clean"], 30)
        self.assertEqual(meta["chembl_release"], "ChEMBL_TEST")
        self.assertEqual(meta["activity_type"], "IC50")
        self.assertFalse(meta["licence_verified"])
        self.assertEqual(meta["raw_sha256"], file_sha256(folder / "chembl_CHEMBL203_raw.jsonl"))
        self.assertEqual(meta["csv_sha256"], file_sha256(folder / "chembl_CHEMBL203.csv"))
        saved = json.loads((folder / "chembl_CHEMBL203.meta.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["n_downloaded"], 30)
        header = (folder / "chembl_CHEMBL203.csv").read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(header, "smiles_std,inchikey,pactivity,n_measurements")

    def test_an_existing_raw_file_is_reused_without_downloading(self):
        folder = Path(tempfile.mkdtemp())
        fc.run_target(self.TARGET, folder, FakeSession([page(self.items())]), min_rows=10)
        again = FakeSession([])   # any data request would raise IndexError
        meta = fc.run_target(self.TARGET, folder, again, min_rows=10)
        self.assertEqual(meta["n_clean"], 30)
        self.assertTrue(all(call[0].endswith("status.json") for call in again.calls))

    def test_refresh_downloads_again(self):
        folder = Path(tempfile.mkdtemp())
        fc.run_target(self.TARGET, folder, FakeSession([page(self.items(30))]), min_rows=10)
        meta = fc.run_target(self.TARGET, folder, FakeSession([page(self.items(20))]), refresh=True, min_rows=10)
        self.assertEqual(meta["n_clean"], 20)

    def test_too_few_clean_molecules_stops_with_a_clear_message(self):
        folder = Path(tempfile.mkdtemp())
        with self.assertRaises(fc.DataError) as caught:
            fc.run_target(self.TARGET, folder, FakeSession([page(self.items(5))]), min_rows=50)
        self.assertIn("EGFR", str(caught.exception))
        self.assertFalse((folder / "chembl_CHEMBL203.csv").exists())

    def test_raw_file_round_trip(self):
        path = Path(tempfile.mkdtemp()) / "raw.jsonl"
        sha = fc.write_raw([record("CCO"), record("CCN")], path)
        self.assertEqual(sha, file_sha256(path))
        self.assertEqual(len(fc.read_raw(path)), 2)


if __name__ == "__main__":
    unittest.main()
