"""Train the ML-tier models, then write model files, reference data and model cards (MSP-22).

Run from the repository root, for example:

  python -m training.export_models \
      --solubility-data training/data/aqsoldb.csv --solubility-source aqsoldb \
      --tox21-data training/data/tox21.csv.gz

Writes (models/ is git-ignored, the cards are committed):
  models/ml/*.json          XGBoost native model files (never pickle, SEC-008)
  models/ml/*_train_fp_*.npy, *_descriptor_ranges_*.json   reference data for MSP-24
  models/ml/manifest.json   file name + SHA-256 of everything above
  backend/registry/cards/ml_solubility_xgb_<version>.json, ml_tox21_xgb_<version>.json
"""
import argparse
import platform
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import rdkit
import sklearn
import xgboost as xgb
from rdkit import Chem  # pylint: disable=no-name-in-module

from backend.workers.ml import featuriser
from training import datasets, qa_acc
from training.splits import DEFAULT_FRACTIONS, scaffold_split
from training.train_solubility import SEED, train_solubility
from training.train_tox21 import train_tox21
from training.utils import file_sha256, utc_now_iso, write_json

MODEL_VERSION = "1.0.0"
CARD_VERSION = 1


def build_features(smiles: List[str]) -> np.ndarray:
    """Feature matrix for already-standardised SMILES (same featuriser as serving)."""
    mols = [Chem.MolFromSmiles(value) for value in smiles]  # pylint: disable=no-member
    return featuriser.featurise_mols(mols)


def software_versions() -> Dict[str, str]:
    """Versions recorded in every card, so a result can be reproduced."""
    return {
        "python": platform.python_version(),
        "rdkit": rdkit.__version__,
        "xgboost": xgb.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def featuriser_info() -> Dict:
    """Which features the models were trained on (serving checks the signature)."""
    return {
        "version": featuriser.FEATURISER_VERSION,
        "signature": featuriser.feature_signature(),
        "n_features": featuriser.N_FEATURES,
        "n_descriptors": featuriser.N_DESCRIPTORS,
        "morgan_bits": featuriser.MORGAN_BITS,
        "maccs_bits": featuriser.MACCS_BITS,
    }


def _artifact(path: Path, **extra) -> Dict:
    return {"file": path.name, "sha256": file_sha256(path), "size_bytes": path.stat().st_size, **extra}


def save_model(model, path: Path, probe: np.ndarray, classifier: bool) -> Dict:
    """Save as native XGBoost JSON, reload it, and check the predictions match."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(path))
    booster = xgb.Booster()
    booster.load_model(str(path))
    expected = model.predict_proba(probe)[:, 1] if classifier else model.predict(probe)
    actual = booster.predict(xgb.DMatrix(probe))
    if not np.allclose(expected, actual, atol=1e-5):
        raise RuntimeError(f"{path.name}: reloaded model predicts differently from the trained one")
    return _artifact(path)


def save_reference_data(features: np.ndarray, train_idx: np.ndarray, stem: str, out_dir: Path) -> List[Dict]:
    """Training-set Morgan fingerprints (packed, no pickle) and descriptor ranges for MSP-24."""
    rows = features[train_idx]
    fingerprints = np.packbits(rows[:, featuriser.MORGAN_SLICE].astype(np.uint8), axis=1)
    fp_path = out_dir / f"{stem}_train_fp_{MODEL_VERSION}.npy"
    np.save(fp_path, fingerprints, allow_pickle=False)

    descriptors = rows[:, : featuriser.N_DESCRIPTORS].astype(float)
    ranges = {
        name: {
            "min": float(np.nanmin(descriptors[:, i])),
            "p01": float(np.nanpercentile(descriptors[:, i], 1)),
            "p99": float(np.nanpercentile(descriptors[:, i], 99)),
            "max": float(np.nanmax(descriptors[:, i])),
        }
        for i, name in enumerate(featuriser.DESCRIPTOR_NAMES)
        if not np.all(np.isnan(descriptors[:, i]))
    }
    range_path = out_dir / f"{stem}_descriptor_ranges_{MODEL_VERSION}.json"
    write_json(range_path, ranges)
    return [
        _artifact(fp_path, kind="reference_data", description="packed Morgan bits of the training rows"),
        _artifact(range_path, kind="reference_data", description="descriptor min/p01/p99/max of the training rows"),
    ]


def _data_block(source_key: str, stats: Dict, raw_info: Dict) -> Dict:
    info = datasets.SOURCE_INFO[source_key]
    return {
        "source": info["name"],
        "citation": info["citation"],
        "origin": info["origin"],
        "licence_note": info["licence_note"],
        "licence_verified": False,
        "raw_file": raw_info.get("file"),
        "raw_sha256": raw_info.get("sha256"),
        **stats,
    }


def _split_block(sizes: Dict) -> Dict:
    return {
        "method": "scaffold_split_bemis_murcko",
        "acyclic_molecules": "each its own group, spread across the parts",
        "seed": SEED,
        "fractions": list(DEFAULT_FRACTIONS),
        **sizes,
    }


def export_solubility(clean: pd.DataFrame, stats: Dict, source_key: str, raw_info: Dict, dirs: Dict,
                      params: Optional[Dict] = None, baseline_params: Optional[Dict] = None,
                      n_jobs: int = -1) -> Dict:  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Train the LogS model and write its file, reference data and card."""
    smiles = clean["smiles_std"].tolist()
    features = build_features(smiles)
    target = clean["logs"].to_numpy(dtype=float)
    split = scaffold_split(smiles, seed=SEED)
    result = train_solubility(features, target, split, params, baseline_params, SEED, n_jobs)

    name = "ml_solubility_xgb"
    model_path = dirs["models"] / f"{name}_{MODEL_VERSION}.json"
    artifacts = [save_model(result.model, model_path, features[split[2][:50]], classifier=False)]
    artifacts += save_reference_data(features, split[0], "ml_solubility", dirs["models"])

    checks = qa_acc.solubility_checks(
        result.metrics_test["rmse"], result.metrics_baseline_test["rmse"], result.uncertainty["test_coverage"]
    )
    card = {
        "card_version": CARD_VERSION,
        "name": name,
        "version": MODEL_VERSION,
        "tier": "ml",
        "task": "solubility_logs",
        "requirements": ["FR-010", "FR-012", "QA-ACC", "BR-009"],
        "unit": "log10(mol/L)",
        "trained_at": utc_now_iso(),
        "seed": SEED,
        "software": software_versions(),
        "featuriser": featuriser_info(),
        "data": _data_block(source_key, stats, raw_info),
        "split": _split_block(result.split_sizes),
        "model": {"algorithm": "xgboost.XGBRegressor", "params": result.params, "tuned": False},
        "baseline": {"algorithm": "sklearn.ensemble.RandomForestRegressor", "params": result.baseline_params,
                     "exported": False},
        "metrics": {"test": result.metrics_test, "baseline_test": result.metrics_baseline_test},
        "uncertainty": result.uncertainty,
        "qa_acc": {"checks": checks, "shortfalls": qa_acc.shortfalls(checks)},
        "artifacts": artifacts,
    }
    write_json(dirs["cards"] / f"{name}_{MODEL_VERSION}.json", card)
    return card


def export_tox21(clean: pd.DataFrame, stats: Dict, raw_info: Dict, dirs: Dict,
                 params: Optional[Dict] = None, n_jobs: int = -1) -> Dict:  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Train the 12 Tox21 models and write their files, reference data and card."""
    smiles = clean["smiles_std"].tolist()
    features = build_features(smiles)
    tasks = list(datasets.TOX21_TASKS)
    labels = clean[tasks].to_numpy(dtype=float)
    split = scaffold_split(smiles, seed=SEED)
    result = train_tox21(features, labels, tasks, split, params, SEED, n_jobs)

    artifacts: List[Dict] = []
    task_cards: Dict[str, Dict] = {}
    for task, res in result.tasks.items():
        entry: Dict = {"status": res.status, "n_labelled": res.n_labelled}
        if res.model is not None:
            slug = task.lower().replace("-", "_")
            path = dirs["models"] / f"ml_tox21_{slug}_{MODEL_VERSION}.json"
            file_info = save_model(res.model, path, features[split[2][:50]], classifier=True)
            artifacts.append({**file_info, "task": task})
            entry.update(
                file=path.name,
                calibration=res.platt,
                metrics_test=res.metrics_test,
                metrics_test_uncalibrated=res.metrics_test_uncalibrated,
                reliability_curve=res.reliability_curve,
            )
        task_cards[task] = entry
    artifacts += save_reference_data(features, split[0], "ml_tox21", dirs["models"])

    checks = qa_acc.tox21_checks(result.mean_roc_auc)
    name = "ml_tox21_xgb"
    card = {
        "card_version": CARD_VERSION,
        "name": name,
        "version": MODEL_VERSION,
        "tier": "ml",
        "task": "tox21_assay_signals",
        "requirements": ["FR-011", "QA-ACC", "BR-009"],
        "unit": "probability (0 to 1), calibrated",
        "trained_at": utc_now_iso(),
        "seed": SEED,
        "software": software_versions(),
        "featuriser": featuriser_info(),
        "data": _data_block("tox21", stats, raw_info),
        "split": _split_block(result.split_sizes),
        "model": {"algorithm": "xgboost.XGBClassifier (one per assay)", "params": result.params, "tuned": False},
        "metrics": {"mean_roc_auc_test": result.mean_roc_auc},
        "tasks": task_cards,
        "qa_acc": {"checks": checks, "shortfalls": qa_acc.shortfalls(checks)},
        "artifacts": artifacts,
    }
    write_json(dirs["cards"] / f"{name}_{MODEL_VERSION}.json", card)
    return card


def write_manifest(cards: List[Dict], out_dir: Path) -> Dict:
    """models/ml/manifest.json: every file with its SHA-256, for registering in the models table."""
    entries = []
    for card in cards:
        for item in card["artifacts"]:
            entries.append({**item, "tier": "ml", "model_name": card["name"], "version": card["version"]})
    manifest = {"manifest_version": 1, "generated_at": utc_now_iso(), "files": entries}
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def _raw_info(path: Path) -> Dict:
    return {"file": path.name, "sha256": file_sha256(path)}


def main(argv=None) -> None:  # pragma: no cover - command line wrapper
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--solubility-data", type=Path, required=True)
    parser.add_argument("--solubility-source", choices=["aqsoldb", "esol"], default="aqsoldb")
    parser.add_argument("--tox21-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("models/ml"))
    parser.add_argument("--cards", type=Path, default=Path("backend/registry/cards"))
    parser.add_argument("--jobs", type=int, default=-1, help="CPU threads for XGBoost (-1 = all)")
    args = parser.parse_args(argv)

    dirs = {"models": args.out, "cards": args.cards}
    cards = []

    print("Loading and standardising solubility data ...", flush=True)
    loader = datasets.load_aqsoldb if args.solubility_source == "aqsoldb" else datasets.load_esol
    clean, stats = datasets.clean_regression(loader(args.solubility_data))
    print("  ", stats, flush=True)
    print("Training solubility model ...", flush=True)
    card = export_solubility(clean, stats, args.solubility_source, _raw_info(args.solubility_data), dirs, n_jobs=args.jobs)
    print("  test:", card["metrics"]["test"], "\n  baseline:", card["metrics"]["baseline_test"],
          "\n  interval:", card["uncertainty"], flush=True)
    cards.append(card)

    print("Loading and standardising Tox21 data ...", flush=True)
    clean, stats = datasets.clean_tox21(datasets.load_tox21(args.tox21_data))
    print("  ", stats, flush=True)
    print("Training the 12 Tox21 models ...", flush=True)
    card = export_tox21(clean, stats, _raw_info(args.tox21_data), dirs, n_jobs=args.jobs)
    print("  mean ROC-AUC:", card["metrics"]["mean_roc_auc_test"], flush=True)
    cards.append(card)

    write_manifest(cards, args.out)
    for card in cards:
        print(card["name"], "QA-ACC shortfalls:", card["qa_acc"]["shortfalls"] or "none")
    print("Done. Model files and manifest:", args.out, "| cards:", args.cards)


if __name__ == "__main__":  # pragma: no cover
    main()
