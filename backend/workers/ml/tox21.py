"""12 calibrated Tox21 assay signals.

Owner: Nishitha. Requirements: FR-011, BR-006, NFR-009, NFR-011.

One XGBoost classifier per assay. Each raw probability goes through Platt scaling with
the two numbers (a, b) stored in the model card: sigmoid(a * logit(p) + b). Every value
is labelled `assay_signal`: it is a signal from one Tox21 assay, not a toxicity verdict.

The Platt maths is repeated here on purpose. training/uncertainty.py needs scikit-learn,
which is a development-only package, so the serving code must not import it. A unit test
checks that both give the same numbers.
"""
import math
from typing import Dict, Tuple

import numpy as np
import xgboost as xgb

from backend.core import labels
from backend.registry.loader import ModelLoadError
from backend.workers.ml import model_cards

TASK = "tox21_assay_signals"
CARD_NAME = "ml_tox21_xgb"
PLATT_METHOD = "platt_logit"
_EPS = 1e-6  # same clipping as training/uncertainty.py

ASSAYS = (
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
)


def apply_platt(probability: float, a: float, b: float) -> float:
    """Calibrated probability from a raw model probability."""
    clipped = min(max(float(probability), _EPS), 1.0 - _EPS)
    logit = math.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + math.exp(-(a * logit + b)))


def _calibration(entry: dict) -> Tuple[float, float]:
    calibration = entry.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("method") != PLATT_METHOD:
        raise ModelLoadError("Tox21 card has an unknown calibration method.")
    a, b = calibration.get("a"), calibration.get("b")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (a, b)):
        raise ModelLoadError("Tox21 card has invalid calibration numbers.")
    return float(a), float(b)


class Tox21Models:
    """The verified, loaded assay models with their calibration numbers."""

    def __init__(self, card: dict, boosters: Dict[str, xgb.Booster], files: model_cards.VerifiedFiles):
        self.card = card
        self.boosters = boosters
        self.files = files
        tasks = card["tasks"]
        self.calibration = {assay: _calibration(tasks[assay]) for assay in boosters}

    @classmethod
    def load(cls, card_file=None, folder=None) -> "Tox21Models":
        """Read the card, verify every assay file and load them."""
        card = model_cards.read_card(card_file or model_cards.card_path(CARD_NAME), TASK)
        tasks = card.get("tasks")
        if not isinstance(tasks, dict) or set(tasks) != set(ASSAYS):
            raise ModelLoadError("Tox21 card does not describe the 12 expected assays.")
        trained = {assay: entry for assay, entry in tasks.items() if entry.get("status") == "trained"}
        if not trained:
            raise ModelLoadError("Tox21 card has no trained assay.")
        names = [entry.get("file") for entry in trained.values()]
        if not all(isinstance(name, str) for name in names) or len(set(names)) != len(names):
            raise ModelLoadError("Tox21 card has missing or repeated assay files.")
        verified = model_cards.verify_files(card, names, folder or model_cards.models_folder())
        boosters = {
            assay: model_cards.load_booster(verified.paths[entry["file"]], card)
            for assay, entry in trained.items()
        }
        return cls(card, boosters, verified)

    @property
    def version(self) -> str:
        """Model version from the card."""
        return self.card["version"]

    def predict(self, features: np.ndarray) -> Dict:
        """The 12 assay signals for one feature vector, in the card's assay order."""
        matrix = xgb.DMatrix(np.asarray(features, dtype=np.float32).reshape(1, -1))
        signals = {}
        for assay in ASSAYS:
            booster = self.boosters.get(assay)
            if booster is None:
                signals[assay] = {"probability": None, "status": "unavailable",
                                  "label": labels.ASSAY_SIGNAL}
                continue
            a, b = self.calibration[assay]
            raw = float(booster.predict(matrix)[0])
            signals[assay] = {"probability": apply_platt(raw, a, b), "label": labels.ASSAY_SIGNAL}
        return {"signals": signals, "model_name": self.card["name"], **labels.tox21_labels(self.version)}

    def hash_inputs(self) -> Dict:
        """Everything about these models that affects their output (goes into params_hash)."""
        return {**model_cards.card_identity(self.card),
                "calibration": {assay: list(pair) for assay, pair in sorted(self.calibration.items())}}
