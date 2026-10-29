import json
import tempfile
import unittest
from pathlib import Path

from backend.registry.loader import ModelLoadError, load_model, sha256_of


class LoaderTests(unittest.TestCase):  # SEC-008
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.models = Path(self.tmp.name, "models")
        self.models.mkdir()
        self.cards = Path(self.tmp.name, "cards")
        self.cards.mkdir()

    def make(self, artifact="model.json", content=b'{"weights": [1, 2, 3]}', **overrides):
        (self.models / artifact).write_bytes(content)
        card = {"name": "solubility", "tier": "ml", "version": "1.0", "artifact": artifact,
                "sha256": sha256_of(self.models / artifact)}
        card.update(overrides)
        card_path = self.cards / "card.json"
        card_path.write_text(json.dumps(card), encoding="utf-8")
        return card_path

    def test_good_model_loads(self):
        handle = load_model(self.make(), self.models)
        self.assertEqual((handle.name, handle.tier, handle.version), ("solubility", "ml", "1.0"))
        self.assertTrue(handle.path.is_file())

    def test_wrong_sha256_is_refused(self):
        with self.assertRaises(ModelLoadError):
            load_model(self.make(sha256="0" * 64), self.models)

    def test_pickle_suffix_is_refused(self):
        with self.assertRaises(ModelLoadError):
            load_model(self.make(artifact="model.pkl"), self.models)

    def test_pickle_header_is_refused_even_with_safe_name(self):
        with self.assertRaises(ModelLoadError):
            load_model(self.make(content=b"\x80\x04\x95rest"), self.models)

    def test_path_outside_models_folder_is_refused(self):
        (Path(self.tmp.name) / "outside.json").write_bytes(b"{}")
        card = self.make(artifact="model.json", artifact_path_unused=True)
        data = json.loads(card.read_text(encoding="utf-8"))
        data["artifact"] = "../outside.json"
        data["sha256"] = sha256_of(Path(self.tmp.name) / "outside.json")
        card.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ModelLoadError):
            load_model(card, self.models)

    def test_missing_file_bad_card_and_unknown_tier_are_refused(self):
        card = self.make()
        (self.models / "model.json").unlink()
        with self.assertRaises(ModelLoadError):
            load_model(card, self.models)
        with self.assertRaises(ModelLoadError):
            load_model(self.cards / "nope.json", self.models)
        with self.assertRaises(ModelLoadError):
            load_model(self.make(tier="gpu"), self.models)


if __name__ == "__main__":
    unittest.main()
    