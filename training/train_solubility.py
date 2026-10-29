"""Train the LogS (aqueous solubility) model with a 90% conformal interval (MSP-22).

FR-010 solubility regression, FR-012 90% prediction interval. The model is
gradient-boosted trees (XGBoost); a scikit-learn random forest is trained
alongside only as the QA-ACC baseline and is not exported.

Hyperparameters are plain defaults, not tuned (TBD-C3, provisional).
Run through training.export_models, which also writes the files and the card.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from training.evaluate import regression_metrics
from training.uncertainty import conformal_offset, interval_coverage

SEED = 42
ALPHA = 0.1  # 90% interval

DEFAULT_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 2,
    "subsample": 0.8,
    "colsample_bytree": 0.3,
    "reg_lambda": 1.0,
    "tree_method": "hist",
}

BASELINE_PARAMS = {
    "n_estimators": 300,
    "max_features": 0.3,
    "min_samples_leaf": 1,
}


@dataclass
class SolubilityResult:
    """Everything export_models needs to write files and the model card."""

    model: xgb.XGBRegressor
    params: Dict
    baseline_params: Dict
    metrics_test: Dict
    metrics_baseline_test: Dict
    uncertainty: Dict
    split_sizes: Dict


def train_solubility(
    features: np.ndarray,
    target: np.ndarray,
    split: Tuple[np.ndarray, np.ndarray, np.ndarray],
    params: Optional[Dict] = None,
    baseline_params: Optional[Dict] = None,
    seed: int = SEED,
    n_jobs: int = -1,
) -> SolubilityResult:
    """Fit on the train part, calibrate the interval on the calibration part, score on test."""
    train_idx, cal_idx, test_idx = split
    use = {**DEFAULT_PARAMS, **(params or {})}
    base = {**BASELINE_PARAMS, **(baseline_params or {})}

    model = xgb.XGBRegressor(random_state=seed, n_jobs=n_jobs, **use)
    model.fit(features[train_idx], target[train_idx])
    baseline = RandomForestRegressor(random_state=seed, n_jobs=n_jobs, **base)
    baseline.fit(features[train_idx], target[train_idx])

    pred_cal = model.predict(features[cal_idx])
    pred_test = model.predict(features[test_idx])
    offset = conformal_offset(target[cal_idx], pred_cal, ALPHA)

    return SolubilityResult(
        model=model,
        params=use,
        baseline_params=base,
        metrics_test=regression_metrics(target[test_idx], pred_test),
        metrics_baseline_test=regression_metrics(target[test_idx], baseline.predict(features[test_idx])),
        uncertainty={
            "method": "split_conformal_absolute_residual",
            "alpha": ALPHA,
            "n_calibration": int(len(cal_idx)),
            "offset": offset,
            "n_test": int(len(test_idx)),
            "test_coverage": interval_coverage(target[test_idx], pred_test, offset),
        },
        split_sizes={
            "n_train": int(len(train_idx)),
            "n_calibration": int(len(cal_idx)),
            "n_test": int(len(test_idx)),
        },
    )
