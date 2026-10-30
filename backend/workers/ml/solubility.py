"""LogS prediction with a 90% conformal interval.

Owner: Nishitha. Requirements: FR-010, FR-012, NFR-007, NFR-008.

The interval is split-conformal: prediction +/- the offset stored in the model card
(computed in training/uncertainty.py from held-out calibration residuals). Serving
only reads that number; it never recomputes it.
"""
import math
from typing import Dict

import numpy as np
import xgboost as xgb

from backend.core import labels
from backend.registry.loader import ModelLoadError
from backend.workers.ml import model_cards

TASK = "solubility_logs"
CARD_NAME = "ml_solubility_xgb"


class SolubilityModel:
    """The verified, loaded LogS model and its interval offset."""

    def __init__(self, card: dict, booster: xgb.Booster, files: model_cards.VerifiedFiles):
        self.card = card
        self.booster = booster
        self.files = files
        uncertainty = card.get("uncertainty")
        if not isinstance(uncertainty, dict):
            raise ModelLoadError("Solubility card has no interval information.")
        offset, alpha = uncertainty.get("offset"), uncertainty.get("alpha")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (offset, alpha)):
            raise ModelLoadError("Solubility card has an invalid interval offset.")
        if offset < 0 or not 0 < alpha < 1:
            raise ModelLoadError("Solubility card has an invalid interval offset.")
        self.offset = float(offset)
        self.alpha = float(alpha)

    @classmethod
    def load(cls, card_file=None, folder=None) -> "SolubilityModel":
        """Read the card, verify the model file and load it."""
        card = model_cards.read_card(card_file or model_cards.card_path(CARD_NAME), TASK)
        files = model_cards.model_files(card)
        if len(files) != 1:
            raise ModelLoadError("Solubility card must list exactly one model file.")
        verified = model_cards.verify_files(card, [files[0]["file"]], folder or model_cards.models_folder())
        booster = model_cards.load_booster(next(iter(verified.paths.values())), card)
        return cls(card, booster, verified)

    @property
    def version(self) -> str:
        """Model version from the card."""
        return self.card["version"]

    def predict(self, features: np.ndarray) -> Dict:
        """LogS for one feature vector, with the interval and the label fields."""
        row = np.asarray(features, dtype=np.float32).reshape(1, -1)
        logs = float(self.booster.predict(xgb.DMatrix(row))[0])
        uncertainty = self.card["uncertainty"]
        return {
            "logs": logs,
            "interval": {
                "low": logs - self.offset,
                "high": logs + self.offset,
                "level": 1.0 - self.alpha,
                "method": uncertainty.get("method"),
                "measured_test_coverage": uncertainty.get("test_coverage"),
            },
            "model_name": self.card["name"],
            **labels.solubility_labels(self.version),
        }

    def hash_inputs(self) -> Dict:
        """Everything about this model that affects its output (goes into params_hash)."""
        return {**model_cards.card_identity(self.card),
                "interval_offset": self.offset, "interval_alpha": self.alpha}
