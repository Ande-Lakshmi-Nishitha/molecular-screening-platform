"""GET /api/targets: the supported targets (MSP-26; FR-015, FR-064).

Owner: Nishitha. Needs a live session (login_required). Reads backend/config/targets.yaml, so the
list changes by editing that file, not code. Each entry has the id the submit route accepts as
`target_id`, a display name, the ChEMBL id and the unit (pIC50 or pKi) the DL tier reports.
Does not touch model files: the API container needs no models folder.
Own blueprint, in its own file; registered with one line in api/app.py register_blueprints().
(/api/models is Abheetha's, in api/models_info/routes.py.)
"""
from flask import Blueprint, jsonify

from backend.api.auth.sessions import login_required
from backend.core.config import load_targets

targets_bp = Blueprint("targets", __name__)

PUBLIC_FIELDS = ("id", "name", "chembl_id", "unit")


@targets_bp.get("/api/targets")
@login_required
def list_targets():
    """Every configured target, in the order of the config file."""
    configured = (load_targets() or {}).get("targets") or []
    return jsonify({"targets": [{key: target.get(key) for key in PUBLIC_FIELDS} for target in configured]}), 200
