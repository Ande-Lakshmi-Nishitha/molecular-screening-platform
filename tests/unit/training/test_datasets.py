"""Unit tests for dataset loading and cleaning (MSP-22; BR-001: only valid molecules)."""
# pylint: disable=missing-function-docstring,missing-class-docstring
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from training import datasets


class StandardiseManyTests(unittest.TestCase):
    def test_failures_are_flagged_and_order_is_kept(self):
        result = datasets.standardise_many(["OCC", "not a smiles", "CC(=O)[O-].[Na+]", ""])
        self.assertEqual(result["ok"].tolist(), [True, False, True, False])
        self.assertEqual(result["smiles_std"].tolist()[0], "CCO")
        self.assertEqual(result["smiles_std"].tolist()[2], "CC(=O)O")


class CleanRegressionTests(unittest.TestCase):
    def test_bad_rows_are_dropped_and_duplicates_merged_by_median(self):
        frame = pd.DataFrame(
            {
                "smiles": ["CCO", "OCC", "C(O)C", "c1ccccc1", "garbage", "CC(=O)N"],
                "logs": [-1.0, -2.0, -3.0, -1.5, -4.0, np.nan],
            }
        )
        clean, stats = datasets.clean_regression(frame)
        self.assertEqual(stats["n_raw"], 6)
        self.assertEqual(stats["n_dropped_unparsable"], 1)
        self.assertEqual(stats["n_dropped_missing_value"], 1)
        self.assertEqual(stats["n_duplicates_merged"], 2)
        self.assertEqual(stats["n_clean"], 2)
        by_smiles = dict(zip(clean["smiles_std"], clean["logs"]))
        self.assertEqual(by_smiles["CCO"], -2.0)
        self.assertEqual(by_smiles["c1ccccc1"], -1.5)

    def test_the_same_molecule_after_salt_stripping_is_merged(self):
        frame = pd.DataFrame({"smiles": ["CC(=O)O", "CC(=O)[O-].[Na+]"], "logs": [0.0, 1.0]})
        clean, stats = datasets.clean_regression(frame)
        self.assertEqual(stats["n_clean"], 1)
        self.assertEqual(float(clean["logs"].iloc[0]), 0.5)


class CleanTox21Tests(unittest.TestCase):
    def frame(self, rows):
        data = {"smiles": [row[0] for row in rows]}
        for column, task in enumerate(datasets.TOX21_TASKS):
            data[task] = [row[1][column % len(row[1])] for row in rows]
        return pd.DataFrame(data)

    def test_agreeing_duplicates_keep_their_label(self):
        clean, stats = datasets.clean_tox21(self.frame([("CCO", [1.0]), ("OCC", [1.0])]))
        self.assertEqual(stats["n_clean"], 1)
        self.assertEqual(stats["n_duplicates_merged"], 1)
        self.assertEqual(float(clean["NR-AR"].iloc[0]), 1.0)

    def test_conflicting_duplicates_become_missing(self):
        clean, stats = datasets.clean_tox21(self.frame([("CCO", [1.0]), ("OCC", [0.0])]))
        self.assertTrue(np.isnan(clean["NR-AR"].iloc[0]))
        self.assertEqual(stats["n_conflicting_labels_set_missing"], len(datasets.TOX21_TASKS))

    def test_missing_labels_stay_missing_and_bad_smiles_are_dropped(self):
        clean, stats = datasets.clean_tox21(self.frame([("CCO", [np.nan]), ("garbage", [1.0])]))
        self.assertEqual(stats["n_dropped_unparsable"], 1)
        self.assertTrue(np.isnan(clean["NR-AR"].iloc[0]))


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())

    def test_aqsoldb_columns_are_renamed(self):
        path = self.folder / "aq.csv"
        pd.DataFrame({"SMILES": ["CCO"], "Solubility": [-0.2], "Group": ["G1"]}).to_csv(path, index=False)
        frame = datasets.load_aqsoldb(path)
        self.assertEqual(list(frame.columns), ["smiles", "logs", "group"])
        self.assertEqual(float(frame["logs"].iloc[0]), -0.2)

    def test_esol_columns_are_renamed(self):
        path = self.folder / "esol.csv"
        pd.DataFrame(
            {"smiles": ["CCO"], "measured log solubility in mols per litre": [-0.3]}
        ).to_csv(path, index=False)
        frame = datasets.load_esol(path)
        self.assertEqual(list(frame.columns), ["smiles", "logs"])

    def test_tox21_needs_all_twelve_assays(self):
        path = self.folder / "tox.csv"
        pd.DataFrame({"smiles": ["CCO"], "NR-AR": [1.0]}).to_csv(path, index=False)
        with self.assertRaises(ValueError) as ctx:
            datasets.load_tox21(path)
        self.assertIn("NR-AR-LBD", str(ctx.exception))

    def test_tox21_loads_when_complete(self):
        path = self.folder / "tox_ok.csv"
        data = {"smiles": ["CCO"], **{task: [0.0] for task in datasets.TOX21_TASKS}, "mol_id": ["x"]}
        pd.DataFrame(data).to_csv(path, index=False)
        frame = datasets.load_tox21(path)
        self.assertEqual(list(frame.columns), ["smiles", *datasets.TOX21_TASKS])

    def test_every_source_records_citation_and_licence_note(self):
        for key in ("aqsoldb", "esol", "tox21"):
            with self.subTest(source=key):
                info = datasets.SOURCE_INFO[key]
                self.assertTrue(info["citation"] and info["licence_note"] and info["origin"])
        self.assertEqual(len(datasets.TOX21_TASKS), 12)


if __name__ == "__main__":
    unittest.main()
