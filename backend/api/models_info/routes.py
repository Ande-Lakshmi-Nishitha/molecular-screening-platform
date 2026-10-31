"""GET /api/models: model information for the model page (MSP-38; FR-040, FR-064, BR-009).

Owner: Abheetha. Needs a live session (login_required). /api/targets is in api/targets/routes.py.
Reads the committed model cards in backend/registry/cards, so the page shows exactly what the
workers load, with no database row to keep in step. The API process does not import xgboost or
touch the models folder.

Only a fixed set of card fields is exposed. File names, hashes, requirements, seeds and QA fields
stay out. licence_verified is always present and False unless the card says otherwise, so the page
never presents an unchecked licence as settled.
Register models_bp with one line in api/app.py register_blueprints().
"""
import json
import logging
from pathlib import Path

from flask import Blueprint, jsonify

from backend.api.auth.sessions import login_required

models_bp = Blueprint("models_info", __name__)

CARDS_DIR = Path(__file__).resolve().parents[2] / "registry" / "cards"
TIER_ORDER = ("ml", "dl", "qml")
DATA_FIELDS = ("source", "citation", "origin", "chembl_release", "retrieved_at", "n_clean", "licence_note")
SPLIT_FIELDS = ("method", "n_train", "n_calibration", "n_test")

log = logging.getLogger(__name__)


def _pick(source, fields):
    source = source if isinstance(source, dict) else {}
    return {name: source[name] for name in fields if name in source}


def _scalar(value):
    """A plain value for display: strings and numbers as they are, a dict by its id or name."""
    if isinstance(value, dict):
        value = value.get("id") or value.get("name")
    return value if isinstance(value, (str, int, float)) else None


def _assays(tasks):
    """Names of the assays a multi-task card covers (Tox21)."""
    if isinstance(tasks, dict):
        return [str(name) for name in tasks]
    names = []
    for item in tasks if isinstance(tasks, list) else []:
        name = item if isinstance(item, str) else _scalar(item)
        if name:
            names.append(str(name))
    return names


def model_info(card_id, card):
    """The public fields of one card."""
    data = _pick(card.get("data"), DATA_FIELDS)
    data.setdefault("licence_note", None)
    raw = card.get("data") if isinstance(card.get("data"), dict) else {}
    data["licence_verified"] = raw.get("licence_verified") is True
    return {
        "id": card_id,
        "tier": card.get("tier"),
        "name": card.get("name"),
        "version": card.get("version"),
        "task": card.get("task"),
        "unit": card.get("unit"),
        "target": _scalar(card.get("target")),
        "trained_at": card.get("trained_at"),
        "data": data,
        "split": _pick(card.get("split"), SPLIT_FIELDS),
        "metrics": card.get("metrics") if isinstance(card.get("metrics"), dict) else {},
        "assays": _assays(card.get("tasks")),
    }


def load_model_infos(cards_dir=None):
    """Every readable card as model info, ordered by tier, name and version."""
    infos = []
    for path in sorted(Path(cards_dir or CARDS_DIR).glob("*.json")):
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("Model card %s could not be read.", path.name)
            continue
        if isinstance(card, dict):
            infos.append(model_info(path.stem, card))

    def order(info):
        tier = info["tier"]
        return (TIER_ORDER.index(tier) if tier in TIER_ORDER else len(TIER_ORDER), info["name"] or "", info["version"] or "")

    return sorted(infos, key=order)


@models_bp.get("/api/models")
@login_required
def list_models():
    """Version, training-data source, licence and validation metrics of every model."""
    return jsonify({"models": load_model_infos()}), 200
