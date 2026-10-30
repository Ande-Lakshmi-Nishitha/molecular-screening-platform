"""Job routes: status and partial results (FR-031, FR-032).

Owner: Ayushi. Cancel, retry and history are added in MSP-33 and MSP-37.
"""
import uuid

from flask import Blueprint, g, jsonify

from backend.api.auth.sessions import error_response, login_required
from backend.api.jobs.status import build_status
from backend.db.session import session_scope

jobs_bp = Blueprint("jobs", __name__, url_prefix="/api/jobs")


@jobs_bp.get("/<job_id>")
@login_required
def job_status(job_id):
    """Status and partial results of one of the caller's own jobs. Anything else is a 404."""
    try:
        parsed = uuid.UUID(job_id)
    except ValueError:
        return error_response("JOB_NOT_FOUND", "Job not found.", 404)
    with session_scope() as session:
        body = build_status(session, parsed, uuid.UUID(g.claims["sub"]))
    if body is None:
        return error_response("JOB_NOT_FOUND", "Job not found.", 404)
    return jsonify(body), 200