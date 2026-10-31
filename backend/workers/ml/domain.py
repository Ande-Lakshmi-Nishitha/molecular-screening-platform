"""Applicability-domain check: is this molecule like the training molecules? (MSP-24)

Owner: Nishitha. Requirements: FR-014, NFR-010, SEC-008, TBD-C8.

For each model family the training step wrote two reference files (training/export_models.py),
both listed in the model card as `reference_data` with a SHA-256:

  *_train_fp_<version>.npy              packed 2048-bit Morgan fingerprints of the training rows
  *_descriptor_ranges_<version>.json    min / p01 / p99 / max of each descriptor over those rows

A molecule is inside the domain when BOTH hold:
  1. its highest Tanimoto similarity to any training molecule is at least the threshold
     (AD_TANIMOTO_MIN, 0.3, TBD-C8)
  2. none of the key properties in KEY_DESCRIPTORS lies outside the training min..max
The limits are provisional: they are not tuned on measured errors.

The files go through the same checks as model files (no pickle, SHA-256 must match, plain file name
inside the models folder) and the .npy is read with allow_pickle=False. If they cannot be loaded the
caller reports the domain as 'not checked'; it never reports a made-up 'inside'.
Pure numpy: no Flask, database or Celery imports.
"""
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np

from backend.registry.loader import ModelLoadError
from backend.workers.ml import featuriser, model_cards

FINGERPRINT_MARK = "_train_fp_"
RANGES_MARK = "_descriptor_ranges_"
PACKED_COLUMNS = featuriser.MORGAN_BITS // 8
# Properties that say most about whether a molecule is of a different kind (size, polarity, flexibility).
KEY_DESCRIPTORS = ("MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
                   "HeavyAtomCount")

_POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.int32)


def _reference_names(card: Mapping) -> Dict[str, str]:
    """The card's fingerprint file and ranges file names (exactly one of each)."""
    entries = [entry["file"] for entry in card["artifacts"] if entry.get("kind") == model_cards.REFERENCE_DATA_KIND]
    found = {"fingerprints": [n for n in entries if FINGERPRINT_MARK in n and n.endswith(".npy")],
             "ranges": [n for n in entries if RANGES_MARK in n and n.endswith(".json")]}
    if any(len(names) != 1 for names in found.values()):
        raise ModelLoadError("Model card does not list one fingerprint file and one ranges file.")
    return {key: names[0] for key, names in found.items()}


def _read_ranges(path) -> Dict[str, Dict[str, float]]:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ModelLoadError("Descriptor ranges could not be read.") from exc
    if not isinstance(data, dict):
        raise ModelLoadError("Descriptor ranges must be a JSON object.")
    for name in KEY_DESCRIPTORS:
        entry = data.get(name)
        if not isinstance(entry, dict):
            raise ModelLoadError("Descriptor ranges miss a key property.")
        low, high = entry.get("min"), entry.get("max")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (low, high)) or low > high:
            raise ModelLoadError("Descriptor ranges hold an invalid value.")
    return data


def _read_fingerprints(path) -> np.ndarray:
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ModelLoadError("Training fingerprints could not be read.") from exc
    if array.dtype != np.uint8 or array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != PACKED_COLUMNS:
        raise ModelLoadError("Training fingerprints have an unexpected shape.")
    return array


@dataclass
class DomainReference:
    """Reference data of one model family, verified and loaded."""

    fingerprints: np.ndarray            # packed bits, one row per training molecule
    counts: np.ndarray                  # bits set per training molecule
    ranges: Dict[str, Dict[str, float]]
    files: model_cards.VerifiedFiles
    identity: Dict[str, object]         # goes into params_hash

    @classmethod
    def load(cls, card: Mapping, folder) -> "DomainReference":
        """Verify and load the reference files named by a card (raises ModelLoadError)."""
        names = _reference_names(card)
        verified = model_cards.verify_files(card, list(names.values()), folder)
        fingerprints = _read_fingerprints(verified.paths[names["fingerprints"]])
        ranges = _read_ranges(verified.paths[names["ranges"]])
        sha = {entry["file"]: entry["sha256"] for entry in card["artifacts"]}
        return cls(fingerprints, _POPCOUNT[fingerprints].sum(axis=1), ranges, verified,
                   {"files": {name: sha[name] for name in sorted(names.values())},
                    "key_descriptors": list(KEY_DESCRIPTORS)})

    def max_tanimoto(self, features: np.ndarray) -> float:
        """Highest Tanimoto similarity of a feature vector's Morgan bits to any training molecule."""
        query = np.packbits((np.asarray(features)[featuriser.MORGAN_SLICE] > 0).astype(np.uint8))
        query_count = int(_POPCOUNT[query].sum())
        shared = _POPCOUNT[self.fingerprints & query].sum(axis=1)
        union = self.counts + query_count - shared   # 0 only when both are empty: similarity 0, not 0/0
        return float(np.max(shared / np.maximum(union, 1)))

    def descriptors_outside_range(self, features: np.ndarray) -> List[str]:
        """Key properties that lie outside the min..max of the training molecules."""
        outside = []
        for name in KEY_DESCRIPTORS:
            value = float(features[featuriser.DESCRIPTOR_NAMES.index(name)])
            # a failed descriptor is NaN, and NaN is neither below nor above a bound: it is not judged
            bounds = self.ranges[name]
            if value < bounds["min"] or value > bounds["max"]:
                outside.append(name)
        return outside

    def check(self, features: np.ndarray, threshold: float) -> Dict[str, object]:
        """The domain result for one molecule's feature vector."""
        similarity = self.max_tanimoto(features)
        outside = self.descriptors_outside_range(features)
        return {
            "available": True,
            "in_domain": similarity >= threshold and not outside,
            "max_tanimoto": similarity,
            "threshold": float(threshold),
            "descriptors_outside_range": outside,
        }

    def still_current(self) -> bool:
        """False if a reference file changed on disk since it was verified."""
        return self.files.unchanged()


def unavailable(threshold: float) -> Dict[str, object]:
    """The result when a family's reference data could not be loaded."""
    return {"available": False, "in_domain": None, "max_tanimoto": None,
            "threshold": float(threshold), "descriptors_outside_range": []}


def overall(families: Mapping[str, Mapping[str, object]]) -> Optional[bool]:
    """False if any family is outside, None if one could not be checked (and none is outside), else True."""
    flags = [result["in_domain"] for result in families.values()]
    if False in flags:
        return False
    return None if None in flags else True
