"""QA-ACC checks for the ML tier (thresholds are provisional until TBD-C4 closes).

A model that misses a target is allowed only if the shortfall is reported, so
export_models writes every check, met or not, into the model card.
"""
from typing import Dict, List, Optional

RMSE_VS_BASELINE_FACTOR = 1.10   # about 10% of the Random Forest baseline or better
TOX21_MEAN_AUC_MIN = 0.75
COVERAGE_MIN = 0.85
COVERAGE_MAX = 0.95


def _check(
    check_id: str, criterion: str, target: str, measured: Optional[float], met: bool
) -> Dict:
    return {
        "id": check_id,
        "criterion": criterion,
        "target": target,
        "measured": measured,
        "met": bool(met),
        "provisional": True,
        "reference": "TBD-C4",
    }


def solubility_checks(model_rmse: float, baseline_rmse: float, coverage: float) -> List[Dict]:
    """QA-ACC checks for the solubility model."""
    limit = RMSE_VS_BASELINE_FACTOR * baseline_rmse
    return [
        _check(
            "QA-ACC-ML-RMSE",
            "scaffold-split RMSE within about 10% of the Random Forest baseline or better",
            f"<= {limit:.3f} (1.10 x baseline {baseline_rmse:.3f})",
            model_rmse,
            model_rmse <= limit,
        ),
        _check(
            "QA-ACC-ML-COVERAGE",
            "empirical coverage of the 90% interval",
            f"{COVERAGE_MIN} to {COVERAGE_MAX}",
            coverage,
            COVERAGE_MIN <= coverage <= COVERAGE_MAX,
        ),
    ]


def tox21_checks(mean_auc: Optional[float]) -> List[Dict]:
    """QA-ACC check for the Tox21 models."""
    met = mean_auc is not None and mean_auc >= TOX21_MEAN_AUC_MIN
    return [
        _check(
            "QA-ACC-TOX21-AUC",
            "mean ROC-AUC over the Tox21 assays, scaffold split",
            f">= {TOX21_MEAN_AUC_MIN}",
            mean_auc,
            met,
        )
    ]


def shortfalls(checks: List[Dict]) -> List[str]:
    """Ids of the checks that were not met (these must be reported, not hidden)."""
    return [check["id"] for check in checks if not check["met"]]
