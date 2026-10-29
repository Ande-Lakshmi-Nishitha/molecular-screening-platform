"""Train the 12 Tox21 assay-signal classifiers with calibrated probabilities (MSP-22).

FR-011: one calibrated probability per assay endpoint, presented as an
*assay signal*, not a toxicity assessment. One XGBoost classifier per assay,
trained on the rows that have a label for that assay. Probabilities are
calibrated with Platt scaling fitted on the calibration part (a, b are stored
as plain numbers in the model card).

Hyperparameters are plain defaults, not tuned (TBD-C3, provisional).
Run through training.export_models, which also writes the files and the card.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import xgboost as xgb

from training.evaluate import classification_metrics, mean_of
from training.uncertainty import apply_platt, fit_platt, reliability_table

SEED = 42

DEFAULT_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.3,
    "tree_method": "hist",
    "eval_metric": "logloss",
}


@dataclass
class TaskResult:
    """One assay: the fitted model, its calibration numbers and test metrics."""

    task: str
    model: Optional[xgb.XGBClassifier]
    status: str                      # "trained" or a reason it was skipped
    platt: Dict[str, float] = field(default_factory=dict)
    metrics_test: Dict = field(default_factory=dict)
    metrics_test_uncalibrated: Dict = field(default_factory=dict)
    reliability_curve: list = field(default_factory=list)
    n_labelled: Dict[str, int] = field(default_factory=dict)


@dataclass
class Tox21Result:
    """All assays plus the mean ROC-AUC used by QA-ACC."""

    tasks: Dict[str, TaskResult]
    params: Dict
    mean_roc_auc: Optional[float]
    split_sizes: Dict


def _labelled(labels: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Subset of idx that has a 0/1 label (not NaN)."""
    return idx[~np.isnan(labels[idx])]


def train_task(
    task: str,
    features: np.ndarray,
    labels: np.ndarray,
    split: Tuple[np.ndarray, np.ndarray, np.ndarray],
    params: Dict,
    seed: int,
    n_jobs: int,
) -> TaskResult:
    """Train and calibrate one assay."""
    train_idx, cal_idx, test_idx = (_labelled(labels, part) for part in split)
    counts = {"train": int(len(train_idx)), "calibration": int(len(cal_idx)), "test": int(len(test_idx))}
    if len(np.unique(labels[train_idx])) < 2:
        return TaskResult(task, None, "skipped: training labels have a single class", n_labelled=counts)

    model = xgb.XGBClassifier(random_state=seed, n_jobs=n_jobs, **params)
    model.fit(features[train_idx], labels[train_idx].astype(int))

    p_cal = model.predict_proba(features[cal_idx])[:, 1]
    slope, intercept = fit_platt(p_cal, labels[cal_idx])
    p_raw = model.predict_proba(features[test_idx])[:, 1]
    p_cal_test = apply_platt(p_raw, slope, intercept)
    curve, _ = reliability_table(labels[test_idx], p_cal_test)
    return TaskResult(
        task=task,
        model=model,
        status="trained",
        platt={"method": "platt_logit", "a": slope, "b": intercept},
        metrics_test=classification_metrics(labels[test_idx], p_cal_test),
        metrics_test_uncalibrated=classification_metrics(labels[test_idx], p_raw),
        reliability_curve=curve,
        n_labelled=counts,
    )


def train_tox21(
    features: np.ndarray,
    label_matrix: np.ndarray,
    task_names: Sequence[str],
    split: Tuple[np.ndarray, np.ndarray, np.ndarray],
    params: Optional[Dict] = None,
    seed: int = SEED,
    n_jobs: int = -1,
) -> Tox21Result:
    """Train every assay. label_matrix has one column per task, values 0, 1 or NaN."""
    use = {**DEFAULT_PARAMS, **(params or {})}
    results = {
        name: train_task(name, features, label_matrix[:, column], split, use, seed, n_jobs)
        for column, name in enumerate(task_names)
    }
    return Tox21Result(
        tasks=results,
        params=use,
        mean_roc_auc=mean_of(res.metrics_test.get("roc_auc") for res in results.values()),
        split_sizes={
            "n_train": int(len(split[0])),
            "n_calibration": int(len(split[1])),
            "n_test": int(len(split[2])),
        },
    )
