"""Train a five-model GNN ensemble per target, plus a baseline, and write the model files and cards (MSP-25).

Owner: Nishitha. Requirements: FR-017, FR-018, FR-019, FR-020, QA-ACC. Hyperparameters: TBD-C3 (provisional).

Run from the repository root after training.fetch_chembl has produced training/data/chembl_<id>.csv:

  python -m training.train_gnn                 # every target in backend/config/targets.yaml
  python -m training.train_gnn --target EGFR   # one target
  python -m training.train_gnn --quick         # tiny run to check the whole chain works (minutes);
                                               # writes to models/dl_quick, never over the real files

For each target it writes (models/ is git-ignored, the cards are committed):
  models/dl/dl_<target>_gnn_<k>_<version>.ts        five TorchScript models (k = 1..5)
  models/dl/dl_<target>_baseline_<version>.json     native XGBoost baseline (never pickle, SEC-008)
  models/dl/dl_<target>_train_fp_<version>.npy      packed Morgan bits of the training rows (no pickle)
  models/dl/dl_<target>_descriptor_ranges_<version>.json
  models/dl/manifest.json                           file name + SHA-256 of everything above
  backend/registry/cards/dl_<target>_gnn_<version>.json

TorchScript files use the .ts extension on purpose: the model loader refuses .pt and .pth.

Split: the same Bemis-Murcko scaffold split as the ML tier (70 / 10 / 20, seed 42). The ten percent
"calibration" part is used here as the validation set for early stopping; the test part is never seen
during training, and every metric in the card is measured on it. Ensemble members differ only in their
random seed (initial weights and batch order).
"""
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.stats import spearmanr

from backend.core.config import load_yaml_safe
from backend.workers.dl import graph
from training import export_models, qa_acc
from training.evaluate import regression_metrics
from training.fetch_chembl import ACTIVITY_TYPE, DataError
from training.gnn_model import GnnRegressor, to_tensors
from training.splits import scaffold_split
from training.train_solubility import DEFAULT_PARAMS as BASELINE_DEFAULT_PARAMS
from training.utils import file_sha256, utc_now_iso, write_json

SEED = 42
N_MEMBERS = 5
SD_DDOF = 1                  # the SD reported to users is the sample SD of the five members
MODEL_VERSION = export_models.MODEL_VERSION
CARD_VERSION = 1
MODEL_SUFFIX = ".ts"         # not .pt: the loader refuses pickle-style suffixes


@dataclass(frozen=True)
class GnnConfig:
    """GNN size and training settings. Plain defaults, not tuned (TBD-C3)."""

    hidden: int = 128
    layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 80
    patience: int = 12


QUICK_CONFIG = GnnConfig(hidden=32, layers=2, max_epochs=3, patience=2)


def spearman(y_true, y_pred) -> Optional[float]:
    """Spearman rank correlation (None if it cannot be computed)."""
    if len(y_true) < 3 or np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        return None
    return float(spearmanr(y_true, y_pred).statistic)


def score(y_true, y_pred) -> Dict:
    """RMSE, MAE, R^2 and Spearman for one set of predictions."""
    return {**regression_metrics(y_true, y_pred), "spearman": spearman(np.asarray(y_true), np.asarray(y_pred))}


def predict_graphs(model, graphs: Sequence[graph.MolGraph], batch_size: int = 256) -> np.ndarray:
    """Predictions for a list of graphs (model is switched to evaluation mode)."""
    model.eval()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            out.append(model(*to_tensors(list(graphs[start:start + batch_size]))).numpy())
    return np.concatenate(out).astype(float)


def train_member(graphs: Sequence[graph.MolGraph], y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
                 config: GnnConfig, seed: int):  # pylint: disable=too-many-locals
    """Train one network with early stopping on the validation part. Returns (model, info)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    y_train = y[train_idx]
    y_std = float(y_train.std()) or 1.0
    model = GnnRegressor(config.hidden, config.layers, config.dropout, float(y_train.mean()), y_std)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=max(1, config.patience // 3))
    best_rmse, best_state, best_epoch, waited = float("inf"), copy.deepcopy(model.state_dict()), 0, 0
    epochs_run = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        for start in range(0, len(order), config.batch_size):
            chosen = order[start:start + config.batch_size]
            target = torch.from_numpy(y[chosen]).float()
            prediction = model(*to_tensors([graphs[i] for i in chosen]))
            loss = torch.mean(((prediction - target) / y_std) ** 2)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
        epochs_run = epoch
        rmse = float(np.sqrt(np.mean((predict_graphs(model, [graphs[i] for i in val_idx]) - y[val_idx]) ** 2)))
        scheduler.step(rmse)
        if rmse < best_rmse - 1e-4:
            best_rmse, best_state, best_epoch, waited = rmse, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            waited += 1
            if waited >= config.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"seed": seed, "epochs_run": epochs_run, "best_epoch": best_epoch, "validation_rmse": best_rmse}


def save_member(model, path: Path, probe: List[graph.MolGraph]) -> Dict:
    """Export as TorchScript, reload it, and check it predicts the same (also one molecule at a time)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(model.eval()).save(str(path))
    loaded = torch.jit.load(str(path), map_location="cpu")
    loaded.eval()
    with torch.no_grad():
        expected = model(*to_tensors(probe)).numpy()
        actual = loaded(*to_tensors(probe)).numpy()
        single = np.array([float(loaded(*to_tensors([item]))[0]) for item in probe])
    if not (np.allclose(expected, actual, atol=1e-4) and np.allclose(expected, single, atol=1e-4)):
        raise RuntimeError(f"{path.name}: the reloaded TorchScript model predicts differently from the trained one")
    return {"file": path.name, "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def save_baseline(model: xgb.XGBRegressor, path: Path, probe: np.ndarray) -> Dict:
    """Native XGBoost JSON for the baseline (reloaded and checked by export_models.save_model)."""
    return export_models.save_model(model, path, probe, classifier=False)


def software_versions() -> Dict[str, str]:
    """Versions recorded in the card, so a result can be reproduced."""
    return {**export_models.software_versions(), "torch": torch.__version__}


def slug_of(target_id: str) -> str:
    """File-name friendly form of a target id."""
    return "".join(ch if ch.isalnum() else "_" for ch in target_id.lower())


def train_target(frame: pd.DataFrame, target: Dict, meta: Dict, dirs: Dict, config: GnnConfig = GnnConfig(),
                 n_members: int = N_MEMBERS, baseline_params: Optional[Dict] = None,
                 threads: int = 0) -> Dict:  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    """Train the ensemble and baseline for one target and write all files and the card."""
    if threads > 0:
        torch.set_num_threads(threads)
    smiles = frame["smiles_std"].tolist()
    y = frame["pactivity"].to_numpy(dtype=float)
    graphs = [graph.smiles_to_graph(value) for value in smiles]
    features = export_models.build_features(smiles)
    split = scaffold_split(smiles, seed=SEED)
    train_idx, val_idx, test_idx = split
    test_graphs = [graphs[i] for i in test_idx]
    slug = slug_of(target["id"])
    name = f"dl_{slug}_gnn"

    members, member_info, member_predictions = [], [], []
    for position in range(n_members):
        started = time.time()
        model, info = train_member(graphs, y, train_idx, val_idx, config, SEED + position)
        predictions = predict_graphs(model, test_graphs)
        info = {**info, "test": score(y[test_idx], predictions), "seconds": round(time.time() - started, 1)}
        print(f"    member {position + 1}/{n_members}: epochs {info['epochs_run']} (best {info['best_epoch']}), "
              f"test RMSE {info['test']['rmse']:.3f}, {info['seconds']} s", flush=True)
        members.append(model)
        member_info.append(info)
        member_predictions.append(predictions)

    stack = np.vstack(member_predictions)
    mean, sd = stack.mean(axis=0), stack.std(axis=0, ddof=SD_DDOF)
    errors = np.abs(y[test_idx] - mean)
    ensemble_test = score(y[test_idx], mean)

    baseline = xgb.XGBRegressor(random_state=SEED, n_jobs=threads or -1, **{**BASELINE_DEFAULT_PARAMS, **(baseline_params or {})})
    baseline.fit(features[train_idx], y[train_idx])
    baseline_test = score(y[test_idx], baseline.predict(features[test_idx]))

    out = dirs["models"]
    probe = test_graphs[:10]
    artifacts: List[Dict] = []
    for position, model in enumerate(members, start=1):
        path = out / f"dl_{slug}_gnn_{position}_{MODEL_VERSION}{MODEL_SUFFIX}"
        artifacts.append({**save_member(model, path, probe), "role": "gnn_member", "member": position})
    base_path = out / f"dl_{slug}_baseline_{MODEL_VERSION}.json"
    artifacts.append({**save_baseline(baseline, base_path, features[test_idx[:10]]), "role": "baseline"})
    artifacts += export_models.save_reference_data(features, train_idx, f"dl_{slug}", out)

    checks = qa_acc.dl_activity_checks(ensemble_test["rmse"], ensemble_test["spearman"])
    card = {
        "card_version": CARD_VERSION,
        "name": name,
        "version": MODEL_VERSION,
        "tier": "dl",
        "task": "target_activity",
        "requirements": ["FR-015", "FR-016", "FR-017", "FR-018", "FR-019", "FR-020", "QA-ACC", "BR-009"],
        "target": {"id": target["id"], "name": target.get("name"), "chembl_id": target["chembl_id"]},
        "unit": target["unit"],
        "trained_at": utc_now_iso(),
        "seed": SEED,
        "software": software_versions(),
        "graph": graph.graph_info(),
        "featuriser": export_models.featuriser_info(),
        "data": {
            "source": "ChEMBL",
            "citation": "Zdrazil et al., Nucleic Acids Res. 52, D1180 (2024), doi:10.1093/nar/gkad1004",
            **{key: meta.get(key) for key in (
                "origin", "chembl_release", "retrieved_at", "query", "licence_note", "licence_verified",
                "raw_file", "raw_sha256", "csv_sha256", "max_spread", "pactivity_range")},
            **{key: value for key, value in meta.items() if key.startswith("n_")},
        },
        "split": export_models._split_block({  # pylint: disable=protected-access
            "n_train": int(len(train_idx)), "n_calibration": int(len(val_idx)), "n_test": int(len(test_idx)),
            "calibration_part_used_as": "validation set for early stopping",
        }),
        "model": {
            "algorithm": "GINE-style message passing GNN, plain PyTorch, TorchScript",
            "params": asdict(config),
            "tuned": False,
            "ensemble": {"size": n_members, "seeds": [SEED + i for i in range(n_members)], "sd_ddof": SD_DDOF,
                         "members_differ_by": "random seed only"},
            "members": member_info,
        },
        "baseline": {"algorithm": "xgboost.XGBRegressor on the ML featuriser (2431 features)",
                     "params": {**BASELINE_DEFAULT_PARAMS, **(baseline_params or {})}, "exported": True},
        "metrics": {
            "ensemble_test": ensemble_test,
            "baseline_test": baseline_test,
            "ensemble_sd_test": {
                "mean_sd": float(sd.mean()),
                "spearman_sd_vs_abs_error": spearman(errors, sd),
            },
        },
        "qa_acc": {"checks": checks, "shortfalls": qa_acc.shortfalls(checks)},
        "artifacts": artifacts,
    }
    write_json(dirs["cards"] / f"{name}_{MODEL_VERSION}.json", card)
    return card


def read_target_data(target: Dict, data_dir: Path):
    """The cleaned csv and meta file that training.fetch_chembl wrote for one target."""
    csv_path = data_dir / f"chembl_{target['chembl_id']}.csv"
    meta_path = data_dir / f"chembl_{target['chembl_id']}.meta.json"
    if not csv_path.exists() or not meta_path.exists():
        raise DataError(f"{target['id']}: run 'python -m training.fetch_chembl --target {target['id']}' first")
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("csv_sha256") != file_sha256(csv_path):
        raise DataError(f"{csv_path.name} was changed after it was downloaded (SHA-256 differs from the meta file)")
    return pd.read_csv(csv_path), meta


def write_manifest(cards_dir: Path, out_dir: Path) -> Dict:
    """models/dl/manifest.json: every file of every DL card with its SHA-256 (for the models table)."""
    entries = []
    for card_path in sorted(cards_dir.glob(f"dl_*_gnn_{MODEL_VERSION}.json")):
        with open(card_path, encoding="utf-8") as handle:
            card = json.load(handle)
        for item in card["artifacts"]:
            entries.append({**item, "tier": "dl", "model_name": card["name"], "version": card["version"]})
    manifest = {"manifest_version": 1, "generated_at": utc_now_iso(), "files": entries}
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main(argv=None) -> None:  # pragma: no cover - command line wrapper
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--targets-file", type=Path, default=Path("backend/config/targets.yaml"))
    parser.add_argument("--target", help="only this target (id or CHEMBL id)")
    parser.add_argument("--data", type=Path, default=Path("training/data"))
    parser.add_argument("--out", type=Path, default=Path("models/dl"))
    parser.add_argument("--cards", type=Path, default=Path("backend/registry/cards"))
    parser.add_argument("--threads", type=int, default=0, help="CPU threads for torch and XGBoost (0 = default)")
    parser.add_argument("--quick", action="store_true", help="tiny networks, 3 epochs, 2 members: a smoke test only")
    args = parser.parse_args(argv)

    targets = (load_yaml_safe(args.targets_file) or {}).get("targets") or []
    targets = [t for t in targets if not args.target or args.target in (t["id"], t["chembl_id"])]
    if not targets:
        raise SystemExit("No matching target in " + str(args.targets_file))
    for target in targets:
        if target["unit"] not in ACTIVITY_TYPE:
            raise SystemExit(f"{target['id']}: unit must be one of {sorted(ACTIVITY_TYPE)}")
    config, n_members = (QUICK_CONFIG, 2) if args.quick else (GnnConfig(), N_MEMBERS)
    if args.quick:   # never overwrite the real files with a smoke-test run
        args.out, args.cards = args.out.parent / "dl_quick", args.out.parent / "dl_quick" / "cards"
    dirs = {"models": args.out, "cards": args.cards}
    for target in targets:
        print(f"{target['id']} ({target['chembl_id']}, {target['unit']})", flush=True)
        frame, meta = read_target_data(target, args.data)
        print(f"  {len(frame)} molecules; training {n_members} networks and the baseline ...", flush=True)
        card = train_target(frame, target, meta, dirs, config, n_members, threads=args.threads)
        print("  ensemble:", card["metrics"]["ensemble_test"], "\n  baseline:", card["metrics"]["baseline_test"],
              "\n  QA-ACC shortfalls:", card["qa_acc"]["shortfalls"] or "none", flush=True)
        write_manifest(args.cards, args.out)
    print("Done. Model files and manifest:", args.out, "| cards:", args.cards)


if __name__ == "__main__":  # pragma: no cover
    main()
