"""Safe model loader: no pickle, SHA-256 check, model cards (SEC-008).

Owner: Nishitha. Written by Ayushi for MSP-15 (Nishitha reviews).
A model card is a small JSON file in backend/registry/cards/. The model file itself lives in
MODELS_DIR (gitignored). The loader only verifies and locates the file; it never unpickles it.
Card fields: name, tier (ml|dl|qml), version, artifact (file name inside MODELS_DIR), sha256.
"""
import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path

from backend.core.config import Settings

UNSAFE_SUFFIXES = {".pkl", ".pickle", ".joblib", ".pt", ".pth", ".dill", ".cloudpickle"}
REQUIRED_FIELDS = ("name", "tier", "version", "artifact", "sha256")
TIERS = ("ml", "dl", "qml")


class ModelLoadError(Exception):
    """A model file or card is missing, unsafe or fails its hash check."""


@dataclass(frozen=True)
class ModelHandle:
    """A verified model: where it is, which version, and its card."""

    name: str
    tier: str
    version: str
    path: Path
    sha256: str
    card: dict


def sha256_of(path):
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path, expected_sha256):
    """Refuse pickle files and wrong hashes. Returns the verified digest."""
    path = Path(path)
    if path.suffix.lower() in UNSAFE_SUFFIXES:
        raise ModelLoadError(f"Pickle-based model format refused: {path.suffix}")
    if not path.is_file():
        raise ModelLoadError("Model file not found.")
    with open(path, "rb") as handle:
        head = handle.read(2)
    if len(head) == 2 and head[0] == 0x80 and 2 <= head[1] <= 5:
        raise ModelLoadError("File looks like a pickle and was refused.")
    actual = sha256_of(path)
    if not hmac.compare_digest(actual, str(expected_sha256).lower()):
        raise ModelLoadError("SHA-256 mismatch: model file was refused.")
    return actual


def load_card(card_path):
    """Read and check a model card."""
    try:
        with open(card_path, encoding="utf-8") as handle:
            card = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ModelLoadError("Model card could not be read.") from exc
    if not isinstance(card, dict):
        raise ModelLoadError("Model card must be a JSON object.")
    missing = [field for field in REQUIRED_FIELDS if not card.get(field)]
    if missing:
        raise ModelLoadError(f"Model card is missing: {', '.join(missing)}")
    if card["tier"] not in TIERS:
        raise ModelLoadError("Model card has an unknown tier.")
    return card


def load_model(card_path, models_dir=None):
    """Verify the artifact named by the card and return a ModelHandle."""
    base = Path(models_dir or Settings.from_env().models_dir).resolve()
    card = load_card(card_path)
    artifact = (base / card["artifact"]).resolve()
    if base not in artifact.parents:
        raise ModelLoadError("Model file must be inside the models folder.")
    sha = verify_artifact(artifact, card["sha256"])
    return ModelHandle(card["name"], card["tier"], card["version"], artifact, sha, card)
