"""Tests for the GNN model and the DL training / export pipeline (MSP-25; FR-016..FR-020).

Runs on a tiny synthetic set with tiny networks, so it takes seconds. Skipped when torch
is not installed (torch is listed in requirements-dl.txt, not requirements.txt).
"""
# pylint: disable=missing-function-docstring,missing-class-docstring,protected-access,import-outside-toplevel,no-member
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem  # pylint: disable=no-name-in-module

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from backend.workers.dl import graph
from tests.unit.training.synthetic import make_frame

FAST_BASELINE = {"n_estimators": 20, "max_depth": 3}
TARGET = {"id": "TEST_TARGET", "name": "Synthetic target", "chembl_id": "CHEMBL0", "unit": "pIC50"}
META = {"origin": "synthetic", "licence_verified": False, "n_clean": 210, "csv_sha256": "0" * 64}


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def activity_frame():
    frame = make_frame()
    frame["pactivity"] = 6.5 - 0.4 * frame["logs"]
    return frame[["smiles_std", "pactivity"]]


@unittest.skipIf(torch is None, "torch is not installed")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from training import gnn_model
        cls.gnn_model = gnn_model
        cls.smiles = ["CCO", "c1ccccc1O", "CC(=O)Nc1ccccc1", "C", "CCN(CC)CC"]
        cls.graphs = [graph.smiles_to_graph(value) for value in cls.smiles]

    def model(self, **kwargs):
        torch.manual_seed(0)
        return self.gnn_model.GnnRegressor(hidden=16, layers=2, **kwargs).eval()

    def test_one_value_per_molecule(self):
        out = self.model()(*self.gnn_model.to_tensors(self.graphs))
        self.assertEqual(tuple(out.shape), (len(self.smiles),))
        self.assertTrue(torch.isfinite(out).all())

    def test_a_molecule_gives_the_same_value_alone_or_in_a_batch(self):
        model = self.model()
        with torch.no_grad():
            batch = model(*self.gnn_model.to_tensors(self.graphs))
            for position, item in enumerate(self.graphs):
                alone = model(*self.gnn_model.to_tensors([item]))[0]
                self.assertAlmostEqual(float(alone), float(batch[position]), places=4)

    def test_the_answer_does_not_depend_on_atom_order(self):
        model = self.model()
        mol = Chem.MolFromSmiles("CC(=O)Nc1ccccc1O")
        values = []
        for _ in range(5):
            text = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            with torch.no_grad():
                values.append(float(model(*self.gnn_model.to_tensors([graph.smiles_to_graph(text)]))[0]))
        self.assertLess(max(values) - min(values), 1e-4)

    def test_output_is_in_the_targets_own_unit(self):
        plain = self.model(y_mean=0.0, y_std=1.0)
        scaled = self.gnn_model.GnnRegressor(hidden=16, layers=2, y_mean=7.0, y_std=2.0).eval()
        scaled.load_state_dict({**plain.state_dict(), "y_mean": torch.tensor(7.0), "y_std": torch.tensor(2.0)})
        tensors = self.gnn_model.to_tensors(self.graphs)
        np.testing.assert_allclose(scaled(*tensors).detach().numpy(), plain(*tensors).detach().numpy() * 2.0 + 7.0, atol=1e-5)

    def test_a_molecule_without_bonds_works(self):
        out = self.model()(*self.gnn_model.to_tensors([graph.smiles_to_graph("C")]))
        self.assertTrue(torch.isfinite(out).all())

    def test_the_model_can_be_scripted(self):
        scripted = torch.jit.script(self.model())
        tensors = self.gnn_model.to_tensors(self.graphs)
        np.testing.assert_allclose(scripted(*tensors).detach().numpy(), self.model()(*tensors).detach().numpy(), atol=1e-5)


@unittest.skipIf(torch is None, "torch is not installed")
class TrainingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from training import train_gnn
        cls.tg = train_gnn
        cls.folder = Path(tempfile.mkdtemp())
        cls.dirs = {"models": cls.folder / "models" / "dl", "cards": cls.folder / "cards"}
        cls.frame = activity_frame()
        cls.card = train_gnn.train_target(
            cls.frame, TARGET, META, cls.dirs, train_gnn.QUICK_CONFIG, n_members=2, baseline_params=FAST_BASELINE, threads=1
        )

    def test_fr018_the_card_lists_one_torchscript_file_per_member(self):
        members = [a for a in self.card["artifacts"] if a.get("role") == "gnn_member"]
        self.assertEqual([m["member"] for m in members], [1, 2])
        for entry in members:
            self.assertTrue(entry["file"].endswith(".ts"))
            self.assertTrue((self.dirs["models"] / entry["file"]).is_file())

    def test_every_sha256_in_the_card_matches_its_file(self):
        for entry in self.card["artifacts"]:
            with self.subTest(file=entry["file"]):
                self.assertEqual(entry["sha256"], sha256_of(self.dirs["models"] / entry["file"]))

    def test_sec008_no_pickle_suffix_and_no_pickle_header(self):
        for path in self.dirs["models"].iterdir():
            self.assertNotIn(path.suffix, {".pkl", ".pickle", ".pt", ".pth", ".joblib"}, path.name)
            head = path.read_bytes()[:2]
            self.assertFalse(len(head) == 2 and head[0] == 0x80 and 2 <= head[1] <= 5, path.name)

    def test_exported_files_pass_the_platform_loader_checks(self):
        from backend.registry.loader import verify_artifact
        for entry in self.card["artifacts"]:
            verify_artifact(self.dirs["models"] / entry["file"], entry["sha256"])

    def test_exported_torchscript_loads_without_the_training_code_and_matches_the_card_graph(self):
        entry = next(a for a in self.card["artifacts"] if a.get("role") == "gnn_member")
        loaded = torch.jit.load(str(self.dirs["models"] / entry["file"]))
        out = loaded(*self.tg.to_tensors([graph.smiles_to_graph("CCO")]))
        self.assertEqual(tuple(out.shape), (1,))
        self.assertEqual(self.card["graph"]["signature"], graph.graph_signature())

    def test_fr019_baseline_is_a_native_xgboost_file_that_reloads(self):
        import xgboost as xgb
        entry = next(a for a in self.card["artifacts"] if a.get("role") == "baseline")
        self.assertTrue(entry["file"].endswith(".json"))
        booster = xgb.Booster()
        booster.load_model(str(self.dirs["models"] / entry["file"]))
        self.assertGreater(booster.num_features(), 0)

    def test_reference_fingerprints_are_saved_without_pickle(self):
        entry = next(a for a in self.card["artifacts"] if a["file"].endswith("_train_fp_1.0.0.npy"))
        array = np.load(self.dirs["models"] / entry["file"], allow_pickle=False)
        self.assertEqual(array.shape[1], 256)
        self.assertEqual(array.shape[0], self.card["split"]["n_train"])

    def test_fr017_fr020_card_names_target_unit_and_version(self):
        self.assertEqual(self.card["unit"], "pIC50")
        self.assertEqual(self.card["target"]["id"], "TEST_TARGET")
        self.assertEqual(self.card["target"]["chembl_id"], "CHEMBL0")
        self.assertEqual(self.card["version"], "1.0.0")
        self.assertEqual(self.card["tier"], "dl")
        self.assertEqual(self.card["name"], "dl_test_target_gnn")

    def test_metrics_are_measured_on_the_test_part_and_reported_against_the_baseline(self):
        metrics = self.card["metrics"]
        for key in ("rmse", "mae", "r2", "spearman", "n"):
            self.assertIn(key, metrics["ensemble_test"])
            self.assertIn(key, metrics["baseline_test"])
        self.assertEqual(metrics["ensemble_test"]["n"], self.card["split"]["n_test"])
        self.assertIn("mean_sd", metrics["ensemble_sd_test"])

    def test_exported_files_reproduce_the_metrics_in_the_card(self):
        """Load the five-style TorchScript files from disk, redo the test-set maths, compare with the card."""
        from training.splits import scaffold_split
        smiles = self.frame["smiles_std"].tolist()
        test_idx = scaffold_split(smiles, seed=self.tg.SEED)[2]
        y = self.frame["pactivity"].to_numpy(dtype=float)[test_idx]
        graphs = [graph.smiles_to_graph(smiles[i]) for i in test_idx]
        members = [a for a in self.card["artifacts"] if a.get("role") == "gnn_member"]
        with torch.no_grad():
            stack = np.vstack([
                torch.jit.load(str(self.dirs["models"] / m["file"]))(*self.tg.to_tensors(graphs)).numpy() for m in members
            ]).astype(float)
        mean, sd = stack.mean(axis=0), stack.std(axis=0, ddof=1)
        self.assertAlmostEqual(float(np.sqrt(np.mean((mean - y) ** 2))), self.card["metrics"]["ensemble_test"]["rmse"], places=4)
        self.assertAlmostEqual(float(sd.mean()), self.card["metrics"]["ensemble_sd_test"]["mean_sd"], places=4)

    def test_the_ensemble_is_described_so_serving_computes_the_same_sd(self):
        ensemble = self.card["model"]["ensemble"]
        self.assertEqual(ensemble["size"], 2)
        self.assertEqual(ensemble["sd_ddof"], 1)
        self.assertEqual(len(self.card["model"]["members"]), 2)

    def test_qa_acc_checks_are_recorded_even_when_missed(self):
        ids = [c["id"] for c in self.card["qa_acc"]["checks"]]
        self.assertEqual(ids, ["QA-ACC-DL-RMSE", "QA-ACC-DL-SPEARMAN"])
        missed = [c["id"] for c in self.card["qa_acc"]["checks"] if not c["met"]]
        self.assertEqual(self.card["qa_acc"]["shortfalls"], missed)

    def test_the_card_on_disk_is_the_returned_card(self):
        on_disk = json.loads((self.dirs["cards"] / "dl_test_target_gnn_1.0.0.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk, json.loads(json.dumps(self.card)))

    def test_data_block_carries_source_and_licence_status(self):
        data = self.card["data"]
        self.assertEqual(data["source"], "ChEMBL")
        self.assertFalse(data["licence_verified"])

    def test_the_same_seed_gives_the_same_network(self):
        graphs = [graph.smiles_to_graph(value) for value in self.frame["smiles_std"]]
        y = self.frame["pactivity"].to_numpy(dtype=float)
        train_idx, val_idx = np.arange(0, 120), np.arange(120, 160)
        config = self.tg.QUICK_CONFIG
        first, _ = self.tg.train_member(graphs, y, train_idx, val_idx, config, seed=7)
        second, _ = self.tg.train_member(graphs, y, train_idx, val_idx, config, seed=7)
        np.testing.assert_allclose(
            self.tg.predict_graphs(first, graphs[160:170]), self.tg.predict_graphs(second, graphs[160:170]), atol=1e-6
        )

    def test_training_reduces_the_error_compared_with_predicting_the_mean(self):
        graphs = [graph.smiles_to_graph(value) for value in self.frame["smiles_std"]]
        y = self.frame["pactivity"].to_numpy(dtype=float)
        train_idx, val_idx, test_idx = np.arange(0, 140), np.arange(140, 170), np.arange(170, len(y))
        config = self.tg.GnnConfig(hidden=32, layers=2, max_epochs=25, patience=25)
        model, info = self.tg.train_member(graphs, y, train_idx, val_idx, config, seed=1)
        predictions = self.tg.predict_graphs(model, [graphs[i] for i in test_idx])
        model_rmse = float(np.sqrt(np.mean((predictions - y[test_idx]) ** 2)))
        mean_rmse = float(np.sqrt(np.mean((y[train_idx].mean() - y[test_idx]) ** 2)))
        self.assertLess(model_rmse, mean_rmse)
        self.assertGreaterEqual(info["best_epoch"], 1)

    def test_early_stopping_stops_before_the_epoch_limit_when_nothing_improves(self):
        graphs = [graph.smiles_to_graph(value) for value in self.frame["smiles_std"]]
        rng = np.random.default_rng(0)
        y = rng.normal(6, 1, len(graphs))      # pure noise: validation never really improves
        config = self.tg.GnnConfig(hidden=16, layers=1, max_epochs=40, patience=3)
        _, info = self.tg.train_member(graphs, y, np.arange(0, 140), np.arange(140, 180), config, seed=3)
        self.assertLess(info["epochs_run"], 40)


@unittest.skipIf(torch is None, "torch is not installed")
class HelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from training import train_gnn
        cls.tg = train_gnn

    def test_spearman_edge_cases(self):
        self.assertIsNone(self.tg.spearman(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])))
        self.assertIsNone(self.tg.spearman(np.array([1.0, 2.0]), np.array([1.0, 2.0])))
        self.assertAlmostEqual(self.tg.spearman(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 4.0, 6.0, 8.0])), 1.0)

    def test_slug_is_safe_for_file_names(self):
        self.assertEqual(self.tg.slug_of("HIV1_PROTEASE"), "hiv1_protease")
        self.assertEqual(self.tg.slug_of("EGFR (wild type)"), "egfr__wild_type_")

    def test_read_target_data_checks_the_csv_against_the_meta_file(self):
        from training import fetch_chembl
        folder = Path(tempfile.mkdtemp())
        target = {"id": "EGFR", "chembl_id": "CHEMBL203", "unit": "pIC50"}
        with self.assertRaises(fetch_chembl.DataError):
            self.tg.read_target_data(target, folder)          # nothing downloaded yet
        csv_path = folder / "chembl_CHEMBL203.csv"
        csv_path.write_text("smiles_std,inchikey,pactivity,n_measurements\nCCO,X,6.0,1\n", encoding="utf-8")
        from training.utils import file_sha256, write_json
        write_json(folder / "chembl_CHEMBL203.meta.json", {"csv_sha256": file_sha256(csv_path)})
        frame, _ = self.tg.read_target_data(target, folder)
        self.assertEqual(len(frame), 1)
        csv_path.write_text("smiles_std,inchikey,pactivity,n_measurements\nCCO,X,9.9,1\n", encoding="utf-8")
        with self.assertRaises(fetch_chembl.DataError):
            self.tg.read_target_data(target, folder)          # edited after download

    def test_manifest_lists_every_dl_card_in_the_folder(self):
        folder = Path(tempfile.mkdtemp())
        from training.utils import write_json
        for name in ("a", "b"):
            write_json(folder / f"dl_{name}_gnn_1.0.0.json", {
                "name": f"dl_{name}_gnn", "version": "1.0.0", "artifacts": [{"file": f"{name}.ts", "sha256": "0" * 64}]})
        write_json(folder / "ml_other_1.0.0.json", {"name": "x", "version": "1", "artifacts": [{"file": "z"}]})
        manifest = self.tg.write_manifest(folder, folder)
        self.assertEqual([f["file"] for f in manifest["files"]], ["a.ts", "b.ts"])
        self.assertTrue(all(f["tier"] == "dl" for f in manifest["files"]))
        self.assertTrue((folder / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
