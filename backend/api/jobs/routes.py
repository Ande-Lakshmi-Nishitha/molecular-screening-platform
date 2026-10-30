"""Job routes: status, partial results and cancel (FR-031, FR-032, FR-056, FR-057).

Owner: Ayushi. History is added in MSP-37.
"""
import uuid

from flask import Blueprint, g, jsonify
from kombu.exceptions import OperationalError
from redis.exceptions import RedisError

from backend.api.auth.sessions import error_response, login_required
from backend.api.jobs.cancel import cancel_job
from backend.api.jobs.status import JobAccessError, build_status
from backend.celery_app import celery_app
from backend.db.session import session_scope

jobs_bp = Blueprint("jobs", __name__, url_prefix="/api/jobs")


def _parse(job_id):
    try:
        return uuid.UUID(job_id)
    except ValueError:
        return None


def _revoke(task_ids):
    """Ask Celery to drop the cancelled tasks. Best effort: the database already blocks late writes."""
    try:
        for task_id in task_ids:
            celery_app.control.revoke(str(task_id))
    except (OperationalError, RedisError, OSError):
        pass


@jobs_bp.get("/<job_id>")
@login_required
def job_status(job_id):
    """Status and partial results of one of the caller's own jobs (403 for others', 404 if unknown)."""
    parsed = _parse(job_id)
    if parsed is None:
        return error_response("JOB_NOT_FOUND", "Job not found.", 404)
    try:
        with session_scope() as session:
            body = build_status(session, parsed, uuid.UUID(g.claims["sub"]))
    except JobAccessError as err:
        return error_response(err.code, err.message, err.status)
    return jsonify(body), 200


@jobs_bp.delete("/<job_id>")
@login_required
def cancel(job_id):
    """Cancel the caller's job: 403 someone else's, 404 unknown, 409 if every task is final."""
    parsed = _parse(job_id)
    if parsed is None:
        return error_response("JOB_NOT_FOUND", "Job not found.", 404)
    user_id = uuid.UUID(g.claims["sub"])
    try:
        with session_scope() as session:
            cancelled_ids = cancel_job(session, parsed, user_id)
            body = build_status(session, parsed, user_id)
    except JobAccessError as err:
        return error_response(err.code, err.message, err.status)
    _revoke(cancelled_ids)
    return jsonify(body), 200
