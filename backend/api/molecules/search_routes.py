"""GET /api/molecules/search?query=&source=pubchem|chembl (MSP-21; FR-005, FR-006, FR-008, FR-055).

Returns identifiers, structure (SMILES, InChI, image URL) and basic info for up to 5 hits.
A timeout (504), an HTTP error (502) or an empty response (502) returns the standard
error envelope with a message naming the failing service; none of them is cached.
This route is separate from the SMILES submit route, so entering SMILES directly
does not depend on PubChem or ChEMBL being up (FR-008).

Needs a live session (login_required). Own blueprint, in its own file. Register it with
one line in api/app.py register_blueprints().
"""
from flask import Blueprint, current_app, jsonify, request

from backend.api.auth.sessions import login_required
from backend.api.errors import ApiError
from backend.services.search_client.errors import UpstreamError
from backend.services.search_client.service import CLIENTS, MAX_QUERY_LEN, get_default_service

search_bp = Blueprint("molecule_search", __name__)

SERVICE_CONFIG_KEY = "SEARCH_SERVICE"  # tests may set a SearchService here


def _service():
    return current_app.config.get(SERVICE_CONFIG_KEY) or get_default_service()


def _bad_request(field, kind, message):
    return ApiError(400, "validation_error", message, [{"field": field, "type": kind}])


@search_bp.get("/api/molecules/search")
@login_required
def search_molecules():
    """Search PubChem or ChEMBL by compound name."""
    query = (request.args.get("query") or "").strip()
    source = (request.args.get("source") or "pubchem").strip().lower()
    if not query:
        raise _bad_request("query", "missing", "A search query is required.")
    if len(query) > MAX_QUERY_LEN:
        raise _bad_request("query", "string_too_long", f"The query must be at most {MAX_QUERY_LEN} characters.")
    if source not in CLIENTS:
        raise _bad_request("source", "enum", f"The source must be one of: {', '.join(CLIENTS)}.")
    try:
        return jsonify(_service().search(query, source)), 200
    except UpstreamError as exc:
        raise ApiError(exc.status, exc.code, str(exc), [{"service": exc.service, "type": exc.kind}]) from None
