"""Job status and partial results (FR-031, FR-032, FR-036).

Owner: Ayushi. Results keep model_version and params_hash, and the job keeps rules_version,
so every stored answer can be traced to the versions that produced it.
"""
from sqlalchemy import select

from backend.db.models import TIERS, Job, JobTask, Result

ACTIVE = ("queued", "running")


def overall_status(task_statuses, cancelled=False):
    """Roll the task statuses up into one job status."""
    statuses = list(task_statuses)
    if any(s in ACTIVE for s in statuses):
        return "queued" if all(s in ("queued", "skipped") for s in statuses) else "running"
    done = statuses.count("done")
    problems = statuses.count("failed") + statuses.count("cancelled")
    if done == 0:
        return "cancelled" if cancelled or "cancelled" in statuses else "failed"
    return "partial" if problems else "complete"


def _iso(value):
    return value.isoformat() if value else None


def _task_body(task, result):
    body = {"id": str(task.id), "tier": task.tier, "status": task.status,
            "skip_reason": task.skip_reason, "error": task.error, "attempts": task.attempts,
            "started_at": _iso(task.started_at), "finished_at": _iso(task.finished_at),
            "result": None}
    if task.status == "done" and result is not None:
        body["result"] = {"model_version": result.model_version,
                          "params_hash": result.params_hash, "payload": result.payload}
    return body


def build_status(session, job_id, user_id):
    """Return the status body, or None if the job does not exist or belongs to someone else."""
    job = session.get(Job, job_id)
    if job is None or job.user_id != user_id:
        return None
    tasks = session.execute(select(JobTask).where(JobTask.job_id == job.id)).scalars().all()
    tasks.sort(key=lambda task: TIERS.index(task.tier) if task.tier in TIERS else len(TIERS))
    results = {}
    if tasks:
        rows = session.execute(select(Result).where(Result.job_task_id.in_([t.id for t in tasks])))
        results = {row.job_task_id: row for row in rows.scalars()}
    return {
        "job_id": str(job.id),
        "status": overall_status([t.status for t in tasks], job.cancelled_at is not None),
        "target_id": job.target_id,
        "rules_version": job.rules_version,
        "created_at": _iso(job.created_at),
        "finished_at": _iso(job.finished_at),
        "cancelled_at": _iso(job.cancelled_at),
        "summary": job.summary,
        "tasks": [_task_body(task, results.get(task.id)) for task in tasks],
    }
