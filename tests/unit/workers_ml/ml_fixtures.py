"""Real, tiny ML model files and cards for the MSP-23 tests.

Built once per test run with the project's own export code (training.export_models) on a
small synthetic set, so the tests read exactly the card layout production uses. Nothing
here needs the real datasets or the real models/ folder.
"""
import atexit
import shutil
import tempfile
from pathlib import Path
from typing import Dict

from tests.unit.training.synthetic import make_frame
from training import export_models

FAST_REGRESSOR = {"n_estimators": 25, "max_depth": 3}
FAST_BASELINE = {"n_estimators": 15, "max_depth": 4}
FAST_CLASSIFIER = {"n_estimators": 20, "max_depth": 3}
RAW = {"file": "synthetic.csv", "sha256": "0" * 64}
STATS = {"n_raw": 220, "n_clean": 220}

_BUILT: Dict[str, Path] = {}


def built_root() -> Path:
    """Folder holding models/ml/ and cards/ (built on first use, removed at exit)."""
    if "root" not in _BUILT:
        root = Path(tempfile.mkdtemp(prefix="msp23_models_"))
        atexit.register(shutil.rmtree, root, ignore_errors=True)
        dirs = {"models": root / "models" / "ml", "cards": root / "cards"}
        frame = make_frame()
        export_models.export_solubility(frame, STATS, "aqsoldb", RAW, dirs,
                                        FAST_REGRESSOR, FAST_BASELINE, n_jobs=1)
        export_models.export_tox21(frame, STATS, RAW, dirs, FAST_CLASSIFIER, n_jobs=1)
        _BUILT["root"] = root
    return _BUILT["root"]


def copy_of_built(parent: Path) -> Path:
    """A private copy of the built folder, so a test may damage files freely."""
    target = parent / "copy"
    shutil.copytree(built_root(), target)
    return target


def models_dir(root: Path) -> Path:
    """The MODELS_DIR value for a built root (the folder that contains ml/)."""
    return root / "models"


def card_file(root: Path, name: str, version: str = export_models.MODEL_VERSION) -> Path:
    """Path of a card inside a built root."""
    return root / "cards" / f"{name}_{version}.json"
