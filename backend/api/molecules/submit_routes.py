"""Submit route: validate, standardise, create job and task rows, queue the tiers.

Owner: Ayushi. Requirements: FR-029, FR-030, FR-035, SEC-002.
The molecule is checked by validate_smiles(), the same code behind
POST /api/molecules/validate (MSP-20), so the two routes always agree.
A task row is committed before its message is sent, so a worker always finds its row.
"""
import uuid
from datetime import datetime, timezone

from flask import Blueprint, g, jsonify, request
from kombu.exceptions import OperationalError
from redis.exceptions import RedisError

from backend.api.auth.sessions import error_response, login_required
from backend.celery_app import enqueue_task
from backend.core import audit
from backend.core.config import Settings, load_targets, load_triage_rules
from backend.db.models import TIERS
from backend.db.repositories import jobs
from backend.db.session import session_scope
from backend.services.molecule.eligibility import TierDecision
from backend.services.molecule.errors import http_status_for
from backend.services.molecule.limits import MoleculeLimits
from backend.services.molecule.standardise import MoleculeError
from backend.services.molecule.validate import validate_smiles

submit_bp = Blueprint("submit", __name__, url_prefix="/api/molecules")


class SubmitError(Exception):
    """A request problem that maps to the uniform error body."""

    def __init__(self, message, details=None, code="VALIDATION_ERROR", status=422):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.code = code
        self.status = status


def _read_request(settings):
    """Check the request shape. Length and structure are checked by validate_smiles()."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise SubmitError("Request body must be JSON.")
    smiles = data.get("smiles")
    if not isinstance(smiles, str) or not smiles.strip():
        raise SubmitError("A SMILES string is required.", {"field": "smiles"})
    smiles = smiles.strip()
    target_id = data.get("target_id")
    if target_id is not None:
        known = {t.get("id") for t in (load_targets(settings) or {}).get("targets") or []}
        if not isinstance(target_id, str) or target_id not in known:
            raise SubmitError("Unknown target.", {"field": "target_id"})
    tiers = data.get("tiers")
    if tiers is not None and (not isinstance(tiers, list) or not tiers
                              or not set(tiers) <= set(TIERS)):
        raise SubmitError("tiers must be a non-empty list of ml, dl, qml.", {"field": "tiers"})
    return smiles, target_id, tiers


def _molecule_and_tiers(smiles, target_id, requested, settings):
    try:
        std = validate_smiles(smiles, target_id, MoleculeLimits.from_env(settings))
    except MoleculeError as err:
        # 413 INPUT_TOO_LARGE, 422 VALIDATION_ERROR or 422 PARSE_TIMEOUT (SAD section 9.1)
        raise SubmitError(err.message, {"field": "smiles"}, code=err.code,
                          status=http_status_for(err)) from err
    decisions = {tier: TierDecision(**decision) for tier, decision in std.tiers.items()}
    run_tiers = [tier for tier in TIERS if requested is None or tier in requested]
    unavailable = {tier: decisions[tier].skip_reason for tier in run_tiers
                   if requested is not None and not decisions[tier].eligible}
    if unavailable:
        raise SubmitError("A requested tier cannot run for this molecule.",
                          {"tiers": unavailable}, code="TIER_UNAVAILABLE")
    return std, {tier: decisions[tier] for tier in run_tiers}


def _enqueue_all(queued, settings):
    """Send each queued task; on failure mark the unsent ones failed."""
    sent = 0
    try:
        for task_id, tier in queued:
            enqueue_task(task_id, tier, settings)
            sent += 1
    except (OperationalError, RedisError, OSError):
        with session_scope() as session:
            jobs.mark_tasks_failed(session, [task_id for task_id, _ in queued[sent:]], "ENQUEUE_FAILED")
        return False
    return True


@submit_bp.post("/submit")
@login_required
def submit():
    """Create a job for the logged-in user and queue its tiers (202)."""
    settings = Settings.from_env()
    try:
        smiles, target_id, requested = _read_request(settings)
        std, decisions = _molecule_and_tiers(smiles, target_id, requested, settings)
    except SubmitError as err:
        return error_response(err.code, err.message, err.status, err.details)

    user_id = uuid.UUID(g.claims["sub"])
    rules_version = (load_triage_rules(settings) or {}).get("version")
    with session_scope() as session:
        molecule = jobs.get_or_create_molecule(session, std)
        job = jobs.create_job(session, molecule.id, user_id, target_id,
                              {"tiers": list(decisions)}, rules_version)
        tasks = jobs.create_tasks(session, job.id, decisions, datetime.now(timezone.utc))
        audit.record(session, "job_submitted", user_id, {"job_id": str(job.id)})
        job_id = job.id
        rows = [(task.id, task.tier, task.status, task.skip_reason) for task in tasks]

    queued = [(task_id, tier) for task_id, tier, status, _ in rows if status == "queued"]
    if not _enqueue_all(queued, settings):
        return error_response("SERVICE_UNAVAILABLE", "Could not queue the job. Try again.", 503)
    return jsonify({
        "job_id": str(job_id),
        "molecule": {key: value for key, value in std.to_dict().items() if key != "tiers"},
        "tasks": [{"id": str(task_id), "tier": tier, "status": status, "skip_reason": skip}
                  for task_id, tier, status, skip in rows],
    }), 202
