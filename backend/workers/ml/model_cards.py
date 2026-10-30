"""Read and verify ML model cards, then load their XGBoost files (MSP-23; SEC-008).

Owner: Nishitha. Requirements: FR-010, FR-011, FR-012, SEC-008, BR-009.

The ML cards written by training/export_models.py list their files under
`artifacts` (one entry per file: name and SHA-256). The shared loader in
backend/registry/loader.py reads a card with ONE `artifact`, so this module is the
adapter: it applies the same safety rules to every file the ML tier uses, by calling
the loader's own verify_artifact():

* no pickle: refused by file suffix and by file header
* the SHA-256 in the card must match the file, or the file is refused
* a file name from a card must be a plain name and stay inside the models folder
* the card must name the same featuriser the code uses (feature_signature())

Nothing here falls back to another model: any problem raises ModelLoadError, which
the task turns into the task error MODEL_UNAVAILABLE.

Models live in <MODELS_DIR>/ml/ (the folder export_models writes to).
"""
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import xgboost as xgb

from backend.core.config import Settings
from backend.registry.loader import ModelLoadError, verify_artifact
from backend.workers.ml import featuriser

ML_TIER = "ml"
MODELS_SUBFOLDER = "ml"
CARDS_DIR = Path(__file__).resolve().parents[2] / "registry" / "cards"
DEFAULT_MODEL_VERSION = "1.0.0"
MODEL_VERSION_ENV = "ML_MODEL_VERSION"
REFERENCE_DATA_KIND = "reference_data"


def model_version() -> str:
    """Model version to serve; the cards on main are 1.0.0 (override with ML_MODEL_VERSION)."""
    return os.environ.get(MODEL_VERSION_ENV) or DEFAULT_MODEL_VERSION


def card_path(name: str, version: str = "", cards_dir=None) -> Path:
    """Path of the card <name>_<version>.json (the committed cards folder by default)."""
    return Path(cards_dir or CARDS_DIR) / f"{name}_{version or model_version()}.json"


def models_folder(models_dir=None, subfolder: str = MODELS_SUBFOLDER) -> Path:
    """The folder holding a tier's model files (ml/ by default; the DL worker passes "dl")."""
    return Path(models_dir or Settings.from_env().models_dir) / subfolder


@dataclass(frozen=True)
class VerifiedFiles:
    """Model files that passed verification, plus a snapshot to spot later changes."""

    paths: Dict[str, Path]
    snapshot: Tuple[Tuple[str, int, int], ...]

    def unchanged(self) -> bool:
        """True while every file still has the size and modification time it had at load."""
        try:
            return take_snapshot(self.paths.values()) == self.snapshot
        except OSError:
            return False


def take_snapshot(paths) -> Tuple[Tuple[str, int, int], ...]:
    """(path, size, modification time) for each file, in a fixed order."""
    entries = []
    for path in sorted(str(item) for item in paths):
        info = os.stat(path)
        entries.append((path, info.st_size, info.st_mtime_ns))
    return tuple(entries)


def read_card(path, expected_task: str, tier: str = ML_TIER) -> dict:
    """Read a model card (ML by default) and check the fields serving depends on."""
    try:
        with open(path, encoding="utf-8") as handle:
            card = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ModelLoadError("Model card could not be read.") from exc
    if not isinstance(card, dict):
        raise ModelLoadError("Model card must be a JSON object.")
    if card.get("tier") != tier or card.get("task") != expected_task:
        raise ModelLoadError("Model card is not the expected card.")
    for field in ("name", "version"):
        if not isinstance(card.get(field), str) or not card[field]:
            raise ModelLoadError(f"Model card is missing: {field}")
    artifacts = card.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ModelLoadError("Model card lists no files.")
    for entry in artifacts:
        if not isinstance(entry, dict) or not entry.get("file") or not entry.get("sha256"):
            raise ModelLoadError("Model card has a file entry without a name or SHA-256.")
    check_featuriser(card)
    return card


def check_featuriser(card: Mapping) -> None:
    """Refuse a model built with other features than the ones this code produces."""
    info = card.get("featuriser")
    if not isinstance(info, dict):
        raise ModelLoadError("Model card does not record its featuriser.")
    if info.get("signature") != featuriser.feature_signature():
        raise ModelLoadError("Model was built with different features than this code produces.")
    if info.get("n_features") != featuriser.N_FEATURES:
        raise ModelLoadError("Model card has a different feature count than this code produces.")


def model_files(card: Mapping) -> List[dict]:
    """The card's model files (everything except reference data for the domain checker)."""
    return [entry for entry in card["artifacts"] if entry.get("kind") != REFERENCE_DATA_KIND]


def verify_files(card: Mapping, names: List[str], folder) -> VerifiedFiles:
    """Verify the named files of a card against their SHA-256 and return their paths."""
    folder = Path(folder).resolve()
    expected = {entry["file"]: entry["sha256"] for entry in card["artifacts"]}
    paths: Dict[str, Path] = {}
    for name in names:
        if name not in expected:
            raise ModelLoadError("Model card does not list a file it refers to.")
        if Path(name).name != name or name in ("", ".", ".."):
            raise ModelLoadError("Model file name must be a plain file name.")
        path = (folder / name).resolve()
        if folder not in path.parents:
            raise ModelLoadError("Model file must be inside the models folder.")
        verify_artifact(path, expected[name])
        paths[name] = path
    return VerifiedFiles(paths, take_snapshot(paths.values()))


def load_booster(path, card: Mapping) -> xgb.Booster:
    """Load one verified XGBoost JSON file and check it has the expected feature count."""
    booster = xgb.Booster()
    try:
        booster.load_model(str(path))
    except xgb.core.XGBoostError as exc:
        raise ModelLoadError("Model file could not be loaded.") from exc
    if booster.num_features() != card["featuriser"]["n_features"]:
        raise ModelLoadError("Model file has a different feature count than its card.")
    booster.set_param({"nthread": 1})  # one row at a time; workers already run in parallel
    return booster


def card_identity(card: Mapping) -> dict:
    """The parts of a card that identify its models and features (used for params_hash)."""
    return {
        "name": card["name"],
        "version": card["version"],
        "featuriser": card["featuriser"]["signature"],
        "files": {entry["file"]: entry["sha256"] for entry in model_files(card)},
    }


def stable_hash(parts: Mapping) -> str:
    """SHA-256 of a canonical JSON form: the same inputs always give the same hash."""
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
