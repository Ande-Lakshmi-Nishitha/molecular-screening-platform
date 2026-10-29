"""Prediction interval and probability calibration (MSP-22; FR-011, FR-012).

Split-conformal 90% interval: take the absolute residuals of the model on a
held-out calibration set; the interval half-width (the "offset") is their
finite-sample-corrected 90th percentile. Tox21 probabilities are calibrated
with Platt scaling, a two-number logistic fit stored as plain numbers in the
model card (no pickled objects).
"""
import math
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def conformal_offset(y_true, y_pred, alpha: float = 0.1) -> float:
    """Half-width of the (1 - alpha) interval from calibration residuals.

    Uses the k-th smallest absolute residual with k = ceil((n + 1) * (1 - alpha)).
    With too few points for that rank, the largest residual is returned.
    """
    residuals = np.sort(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)))
    count = len(residuals)
    if count == 0:
        raise ValueError("conformal_offset needs at least one calibration point")
    rank = min(math.ceil((count + 1) * (1.0 - alpha)), count)
    return float(residuals[rank - 1])


def interval_coverage(y_true, y_pred, offset: float) -> float:
    """Fraction of points inside prediction +/- offset."""
    errors = np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))
    return float(np.mean(errors <= offset))


def _logit(probabilities) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, float), _EPS, 1.0 - _EPS)
    return np.log(clipped / (1.0 - clipped))


def fit_platt(p_raw, labels) -> Tuple[float, float]:
    """Fit calibrated = sigmoid(a * logit(p_raw) + b). Returns (a, b).

    If the calibration labels hold only one class there is nothing to fit, and
    the identity (1.0, 0.0) is returned.
    """
    labels = np.asarray(labels, int)
    if len(np.unique(labels)) < 2:
        return 1.0, 0.0
    fit = LogisticRegression(C=1e4, solver="lbfgs", max_iter=1000)
    fit.fit(_logit(p_raw).reshape(-1, 1), labels)
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


def apply_platt(p_raw, a: float, b: float) -> np.ndarray:
    """Calibrated probabilities from raw model probabilities."""
    return 1.0 / (1.0 + np.exp(-(a * _logit(p_raw) + b)))


def reliability_table(
    labels, probabilities, n_bins: int = 10
) -> Tuple[List[Dict[str, float]], float]:
    """Equal-count reliability bins and the expected calibration error (ECE)."""
    labels = np.asarray(labels, float)
    probabilities = np.asarray(probabilities, float)
    order = np.argsort(probabilities, kind="stable")
    rows: List[Dict[str, float]] = []
    ece = 0.0
    total = len(labels)
    for chunk in np.array_split(order, min(n_bins, total)):
        if len(chunk) == 0:
            continue
        mean_pred = float(probabilities[chunk].mean())
        frac_pos = float(labels[chunk].mean())
        ece += len(chunk) / total * abs(frac_pos - mean_pred)
        rows.append(
            {"count": int(len(chunk)), "mean_predicted": mean_pred, "fraction_positive": frac_pos}
        )
    return rows, float(ece)
