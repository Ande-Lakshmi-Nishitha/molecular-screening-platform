"""GET /api/jobs: the caller's job history, newest first, paged (MSP-37; FR-049, SEC-002).

Owner: Abheetha. Own blueprint in its own file, so Ayushi's jobs/routes.py is not touched.
Only the caller's jobs are ever returned: the query filters on the user id from the token.
Query string: page (default 1) and page_size (default 20, at most 100); anything else is a 400.
The status of each job is the same roll-up as GET /api/jobs/<id> (overall_status).
Register history_bp with one line in api/app.py register_blueprints().
"""
import uuid

from flask import Blueprint, g, jsonify, request

from backend.api.auth.sessions import error_response, login_required
from backend.api.jobs.status import overall_status
from backend.db.models import TIERS
from backend.db.repositories import job_history
from backend.db.session import session_scope

history_bp = Blueprint("job_history", __name__)

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100


class PagingError(ValueError):
    """A bad page or page_size query parameter."""

    def __init__(self, field, message):
        super().__init__(message)
        self.field = field
        self.message = message


def _whole_number(args, name, default, low, high=None):
    raw = args.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise PagingError(name, f"{name} must be a whole number.") from None
    if value < low or (high is not None and value > high):
        limit = f"between {low} and {high}" if high is not None else f"at least {low}"
        raise PagingError(name, f"{name} must be {limit}.")
    return value


def parse_paging(args):
    """Return (page, page_size) from a query string, or raise PagingError."""
    return (_whole_number(args, "page", 1, 1),
            _whole_number(args, "page_size", DEFAULT_PAGE_SIZE, 1, MAX_PAGE_SIZE))


def _iso(value):
    return value.isoformat() if value else None


def history_item(job, molecule, tasks):
    """One history row: the job, its molecule and the status of each tier."""
    tier_status = {task.tier: task.status for task in tasks}
    return {
        "job_id": str(job.id),
        "status": overall_status([task.status for task in tasks], job.cancelled_at is not None),
        "target_id": job.target_id,
        "created_at": _iso(job.created_at),
        "finished_at": _iso(job.finished_at),
        "molecule": {
            "canonical_smiles": molecule.canonical_smiles,
            "formula": molecule.formula,
            "inchikey": molecule.inchikey,
        },
        "tiers": {tier: tier_status[tier] for tier in TIERS if tier in tier_status},
    }


@history_bp.get("/api/jobs")
@login_required
def list_jobs():
    """The caller's jobs, newest first."""
    try:
        page, page_size = parse_paging(request.args)
    except PagingError as err:
        return error_response("VALIDATION_ERROR", err.message, 400, {"field": err.field})
    user_id = uuid.UUID(g.claims["sub"])
    with session_scope() as session:
        total, rows = job_history.list_jobs_owned(session, user_id, page_size, (page - 1) * page_size)
        items = [history_item(*row) for row in rows]
    return jsonify({"jobs": items, "page": page, "page_size": page_size, "total": total}), 200
