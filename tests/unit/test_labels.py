"""Unit tests for the wording file core/labels.py (MSP-23 first version; MSP-32 extends it).

Covers NFR-007, NFR-008, NFR-009, NFR-011, BR-006.
"""
# pylint: disable=missing-function-docstring,missing-class-docstring
import unittest

from backend.core import labels

FORBIDDEN = ("safe", "non-toxic", "nontoxic", "harmless", "effective", "approved", "cure")


class LabelTests(unittest.TestCase):
    def test_nfr007_nfr008_blocks_carry_unit_limitation_and_model_version(self):
        for block in (labels.solubility_labels("1.0.0"), labels.tox21_labels("1.0.0")):
            with self.subTest(unit=block["unit"]):
                self.assertEqual(labels.missing_labels(block), [])
                self.assertEqual(block["model_version"], "1.0.0")

    def test_dl_blocks_carry_the_targets_own_unit_limitation_and_model_version(self):
        for unit in ("pIC50", "pKi"):
            for block in (labels.activity_labels(unit, "dl_x_gnn@1.0.0"), labels.baseline_labels(unit, "dl_x_gnn@1.0.0")):
                with self.subTest(unit=unit):
                    self.assertEqual(labels.missing_labels(block), [])
                    self.assertEqual(block["unit"], unit)
                    self.assertEqual(block["model_version"], "dl_x_gnn@1.0.0")

    def test_the_dl_limitation_says_the_sd_is_not_the_real_error(self):
        self.assertIn("not the real error", labels.LIMITATION_ACTIVITY)
        self.assertIn("not a measurement", labels.LIMITATION_ACTIVITY)

    def test_br006_the_tox21_block_is_labelled_assay_signal(self):
        self.assertEqual(labels.ASSAY_SIGNAL, "assay_signal")
        self.assertEqual(labels.tox21_labels("1.0.0")["label"], "assay_signal")

    def test_nfr009_the_tox21_limitation_names_what_it_does_not_cover(self):
        for word in ("hERG", "hepatotoxicity", "mutagenicity"):
            self.assertIn(word, labels.LIMITATION_TOX21)

    def test_nfr011_no_sentence_calls_a_molecule_safe_or_effective(self):
        texts = [labels.DISCLAIMER, labels.LIMITATION_SOLUBILITY, labels.LIMITATION_TOX21,
                 labels.LIMITATION_ACTIVITY, labels.LIMITATION_BASELINE]
        for text in texts:
            for word in FORBIDDEN:
                self.assertNotIn(word, text.lower())

    def test_the_disclaimer_says_the_values_are_not_experimental_results(self):
        self.assertIn("not experimental results", labels.DISCLAIMER)

    def test_a_missing_or_empty_field_is_reported(self):
        block = labels.solubility_labels("1.0.0")
        self.assertEqual(labels.missing_labels({**block, "unit": ""}), ["unit"])
        del block["limitation"]
        block["model_version"] = None
        self.assertEqual(labels.missing_labels(block), ["limitation", "model_version"])
        self.assertEqual(labels.missing_labels({}), list(labels.REQUIRED_BLOCK_FIELDS))


if __name__ == "__main__":
    unittest.main()
