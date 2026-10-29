"""Metrics for the model cards (MSP-22; evaluation report D-10 is compiled from them)."""
from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from training.uncertainty import reliability_table


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """RMSE, MAE and R^2 for a regression model."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    residual = y_true - y_pred
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual ** 2) / total) if total > 0 else float("nan"),
    }


def classification_metrics(labels, probabilities) -> Dict[str, Optional[float]]:
    """ROC-AUC, Brier score and ECE. AUC is None if only one class is present."""
    labels = np.asarray(labels, int)
    probabilities = np.asarray(probabilities, float)
    both_classes = len(np.unique(labels)) == 2
    _, ece = reliability_table(labels, probabilities)
    return {
        "n": int(len(labels)),
        "n_positive": int(labels.sum()),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both_classes else None,
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "ece": float(ece),
    }


def mean_of(values: Iterable[Optional[float]]) -> Optional[float]:
    """Mean of the values that are not None (None if there are none)."""
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None
