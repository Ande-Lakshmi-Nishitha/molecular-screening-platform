"""Task state changes that are safe to repeat (FR-056..FR-060, BR-004).

Owner: Ayushi. Each change is one conditional UPDATE, so a repeated message, a redelivery or a
late write from a worker whose task was cancelled finds no matching row and is dropped.
Results are written with an upsert on the task id.
"""
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update

from backend.db.models import JobTask, Result

ACTIVE = ("queued", "running")


def _now():
    return datetime.now(timezone.utc)


def _changed(session, statement):
    return session.execute(statement.execution_options(synchronize_session=False)).rowcount == 1


def claim_task(session, task_id, now=None):
    """Mark a queued or running task as running. Returns the attempt number, or 0 if it was dropped."""
    statement = (update(JobTask).where(JobTask.id == task_id, JobTask.status.in_(ACTIVE))
                 .values(status="running", started_at=now or _now(), attempts=JobTask.attempts + 1))
    if not _changed(session, statement):
        return 0
    return session.execute(select(JobTask.attempts).where(JobTask.id == task_id)).scalar_one()


def requeue_task(session, task_id):
    """Put a running task back to queued, ready for a retry."""
    statement = update(JobTask).where(JobTask.id == task_id, JobTask.status == "running").values(status="queued")
    return _changed(session, statement)


def fail_task(session, task_id, reason, now=None):
    """Mark an unfinished task failed with a short reason such as TIMEOUT or COMPUTE_ERROR."""
    statement = (update(JobTask).where(JobTask.id == task_id, JobTask.status.in_(ACTIVE))
                 .values(status="failed", error=str(reason)[:100], finished_at=now or _now()))
    return _changed(session, statement)


def finish_task(session, task_id, model_version, params_hash, payload, now=None):
    """Mark a running task done and upsert its result. Returns False if the write was dropped."""
    statement = (update(JobTask).where(JobTask.id == task_id, JobTask.status == "running")
                 .values(status="done", finished_at=now or _now()))
    if not _changed(session, statement):
        return False
    existing = session.execute(select(Result).where(Result.job_task_id == task_id)).scalar_one_or_none()
    if existing is None:
        session.add(Result(job_task_id=task_id, model_version=model_version,
                           params_hash=params_hash, payload=payload))
    else:
        existing.model_version = model_version
        existing.params_hash = params_hash
        existing.payload = payload
    return True


def is_cancelled(session, task_id):
    """True if the task was cancelled. Long loops (QML) call this each iteration and stop."""
    status = session.execute(select(JobTask.status).where(JobTask.id == task_id)).scalar_one_or_none()
    return status == "cancelled"


def expire_overdue(session, job_id, settings, now=None, grace_s=10):
    """Fail running tasks that are past their hard time limit with reason TIMEOUT (FR-059).

    A worker killed by its hard limit never reports back, so the status read calls this.
    """
    now = now or _now()
    rows = session.execute(select(JobTask).where(JobTask.job_id == job_id, JobTask.status == "running")).scalars().all()
    expired = []
    for task in rows:
        started = task.started_at
        limit = getattr(settings, f"{task.tier}_hard_limit_s", None)
        if started is None or limit is None:
            continue
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        if now - started > timedelta(seconds=limit + grace_s) and fail_task(session, task.id, "TIMEOUT", now):
            expired.append(task.id)
    if expired:
        session.expire_all()
    return expired
