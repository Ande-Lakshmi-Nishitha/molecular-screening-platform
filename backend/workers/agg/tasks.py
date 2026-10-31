"""finalise(job_id) on queue agg: row lock, idempotent (FR-033, BR-005).

Owner: Ayushi. The job row is locked, and the summary is built only if every task is final and no
summary exists yet, so two workers finishing together make one summary. The status read calls the same
function, so a summary also appears when no worker queued the Celery task.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import select

from backend.celery_app import celery_app
from backend.core.config import load_triage_rules
from backend.db.models import FINAL_STATUSES, Job, JobTask, Result
from backend.db.session import session_scope
from backend.workers.agg import triage

TASK_NAME = "msp.agg.finalise"


def finalise_in_session(session, job_id, rules=None, now=None):
    """Return "missing", "exists", "waiting" or "created"."""
    job = session.execute(select(Job).where(Job.id == job_id).with_for_update()).scalar_one_or_none()
    if job is None:
        return "missing"
    if job.summary:
        return "exists"
    rows = session.execute(select(JobTask, Result).outerjoin(Result, Result.job_task_id == JobTask.id)
                           .where(JobTask.job_id == job.id)).all()
    if not rows or any(task.status not in FINAL_STATUSES for task, _ in rows):
        return "waiting"
    rules = rules or load_triage_rules()
    items = [{"tier": task.tier, "status": task.status, "skip_reason": task.skip_reason,
              "error": task.error, "payload": result.payload if result is not None else None}
             for task, result in rows]
    now = now or datetime.now(timezone.utc)
    job.summary = triage.build_triage(items, rules, now)
    job.rules_version = rules.get("version")
    job.finished_at = now
    session.flush()
    return "created"


@celery_app.task(name=TASK_NAME)
def finalise(job_id):
    """Build the triage summary once, when every task of the job is final."""
    with session_scope() as session:
        return finalise_in_session(session, uuid.UUID(str(job_id)))


def enqueue_finalise(job_id, app=None):
    """Queue finalise(job_id) on the agg queue. Workers call this when a task becomes final."""
    return (app or celery_app).send_task(TASK_NAME, args=[str(job_id)], queue="agg")
