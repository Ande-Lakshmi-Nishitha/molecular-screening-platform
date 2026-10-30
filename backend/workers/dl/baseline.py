"""Gradient-boosted tree baseline on fingerprints and descriptors.

Owner: Nishitha. Requirements: FR-019.

A native XGBoost JSON file (never pickle) trained by training/train_gnn.py on the same data as
the network ensemble and the same features as the ML tier. It is reported beside the ensemble,
in the same unit, so a reader can see whether the networks add anything.
"""
import math
from typing import Mapping

import numpy as np
import xgboost as xgb

from backend.registry.loader import ModelLoadError
from backend.workers.ml import model_cards

BASELINE_ROLE = "baseline"


def baseline_file(card: Mapping) -> str:
    """File name of the card's single baseline model."""
    names = [entry["file"] for entry in card["artifacts"] if entry.get("role") == BASELINE_ROLE]
    if len(names) != 1:
        raise ModelLoadError("Model card must list exactly one baseline file.")
    return names[0]


class BaselineModel:
    """The loaded baseline."""

    def __init__(self, booster: xgb.Booster):
        self.booster = booster

    @classmethod
    def load(cls, path, card: Mapping) -> "BaselineModel":
        """Load a verified file (feature count is checked against the card)."""
        return cls(model_cards.load_booster(path, card))

    def predict(self, features: np.ndarray) -> float:
        """The baseline's value for one feature vector."""
        row = np.asarray(features, dtype=np.float32).reshape(1, -1)
        value = float(self.booster.predict(xgb.DMatrix(row))[0])
        if not math.isfinite(value):
            raise ValueError("The baseline returned a value that is not a number.")
        return value
