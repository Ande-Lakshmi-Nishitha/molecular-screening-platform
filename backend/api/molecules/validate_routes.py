"""POST /api/molecules/validate (MSP-20; FR-001, FR-002, FR-004, FR-053, SEC-004, BR-001).

Standardises one molecule and says which tiers would run for it. The route is
thin on purpose: all the work is in backend.services.molecule.validate, the
same code the submit route (MSP-29) uses, so the two can never disagree.

Two input forms:
  * JSON      {"smiles": "...", "target_id": "..."}       (target_id optional)
  * multipart a "file" part with a MOL/SDF file, plus an optional "target_id"
              form field; only the first record is used.

Errors use the standard envelope. The structure is never echoed back and never
logged (SEC-007): messages are fixed sentences.

Needs a live session (login_required from MSP-17): no token gives 401.

Own blueprint, in its own file, so it does not collide with the submit route in
routes.py (Ayushi). Register it with one line in api/app.py register_blueprints().

TODO(MSP-18): add the API rate limit here when the rate-limit helper is on main.
"""
from typing import Optional

from flask import Blueprint, current_app, jsonify, request
from pydantic import Field

from backend.api.auth.sessions import login_required
from backend.api.errors import ApiError
from backend.api.schemas import StrictModel, validate_body
from backend.services.molecule.errors import (
    CODE_PARSE_TIMEOUT,
    CODE_TOO_LARGE,
    CODE_VALIDATION,
    http_status_for,
)
from backend.services.molecule.limits import MoleculeLimits
from backend.services.molecule.standardise import MoleculeError
from backend.services.molecule.validate import validate_molfile, validate_smiles

validate_bp = Blueprint("molecule_validate", __name__)

MAX_TARGET_ID_LEN = 64
LIMITS_CONFIG_KEY = "MOLECULE_LIMITS"  # tests may set a MoleculeLimits here

_DETAIL_TYPE = {
    CODE_VALIDATION: "invalid_structure",
    CODE_TOO_LARGE: "too_large",
    CODE_PARSE_TIMEOUT: "timeout",
}


class ValidateBody(StrictModel):
    """JSON body of POST /api/molecules/validate."""

    smiles: str
    target_id: Optional[str] = Field(default=None, max_length=MAX_TARGET_ID_LEN)


def _limits() -> MoleculeLimits:
    return current_app.config.get(LIMITS_CONFIG_KEY) or MoleculeLimits.from_env()


def _molecule_error(error: MoleculeError, field: str) -> ApiError:
    """Turn a MoleculeError into the API error (status from the SAD table)."""
    detail = {"field": field, "type": _DETAIL_TYPE.get(error.code, "invalid_structure")}
    return ApiError(http_status_for(error), error.code, error.message, [detail])


def _validate_json():
    body = validate_body(ValidateBody)
    try:
        return validate_smiles(body.smiles, body.target_id, _limits())
    except MoleculeError as exc:
        raise _molecule_error(exc, "smiles") from None


def _validate_upload():
    limits = _limits()
    upload = request.files.get("file")
    if upload is None:
        raise ApiError(
            400, "validation_error", "A MOL/SDF file is required in the 'file' field.",
            [{"field": "file", "type": "missing"}],
        )
    target_id = request.form.get("target_id") or None
    if target_id is not None and len(target_id) > MAX_TARGET_ID_LEN:
        raise ApiError(
            400, "validation_error", "Request failed validation.",
            [{"field": "target_id", "type": "string_too_long"}],
        )
    # Read one byte past the limit: enough to tell "too large" without loading more.
    data = upload.stream.read(limits.max_molfile_bytes + 1)
    try:
        return validate_molfile(data, target_id, limits)
    except MoleculeError as exc:
        raise _molecule_error(exc, "file") from None


@validate_bp.post("/api/molecules/validate")
@login_required
def validate_molecule():
    """Standardise a molecule and report per-tier eligibility."""
    if request.mimetype == "multipart/form-data":
        result = _validate_upload()
    elif request.mimetype == "application/json":
        result = _validate_json()
    else:
        raise ApiError(
            415, "unsupported_media_type",
            "Send JSON, or multipart/form-data with a 'file' part.",
        )
    return jsonify(result.to_dict()), 200
