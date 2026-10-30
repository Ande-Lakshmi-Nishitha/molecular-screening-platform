"""Real, tiny DL model files and cards for the MSP-26 tests.

Built once per test run with the project's own training code (training.train_gnn) on a small
synthetic set: five tiny TorchScript networks plus a baseline per target, written in exactly the
layout production uses. Nothing here needs ChEMBL or the real models/ folder. Needs torch.
"""
import atexit
import shutil
import tempfile
from pathlib import Path
from typing import Dict

from tests.unit.training.synthetic import make_frame

TARGETS = [
    {"id": "TEST_TARGET", "name": "Synthetic target A", "chembl_id": "CHEMBL1", "unit": "pIC50"},
    {"id": "OTHER_TARGET", "name": "Synthetic target B", "chembl_id": "CHEMBL2", "unit": "pKi"},
]
FAST_BASELINE = {"n_estimators": 20, "max_depth": 3}
META = {"origin": "synthetic", "licence_verified": False, "n_clean": 210, "csv_sha256": "0" * 64}

_BUILT: Dict[str, Path] = {}


def built_root() -> Path:
    """Folder holding models/dl/ and cards/ (built on first use, removed at exit)."""
    if "root" not in _BUILT:
        from training import train_gnn   # imported late: it needs torch
        root = Path(tempfile.mkdtemp(prefix="msp26_models_"))
        atexit.register(shutil.rmtree, root, ignore_errors=True)
        dirs = {"models": root / "models" / "dl", "cards": root / "cards"}
        frame = make_frame()
        frame["pactivity"] = 6.5 - 0.4 * frame["logs"]
        frame = frame[["smiles_std", "pactivity"]]
        for target in TARGETS:
            train_gnn.train_target(frame, target, META, dirs, train_gnn.QUICK_CONFIG, n_members=5,
                                   baseline_params=FAST_BASELINE, threads=1)
        _BUILT["root"] = root
    return _BUILT["root"]


def copy_of_built(parent: Path) -> Path:
    """A private copy of the built folder, so a test may damage files freely."""
    target = parent / "copy"
    shutil.copytree(built_root(), target)
    return target


def models_dir(root: Path) -> Path:
    """The MODELS_DIR value for a built root (the folder that contains dl/)."""
    return root / "models"


def card_file(root: Path, name: str, version: str = "1.0.0") -> Path:
    """Path of a card inside a built root."""
    return root / "cards" / f"{name}_{version}.json"
