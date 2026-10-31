"""Job status and partial results (FR-031, FR-032, FR-036, FR-059, SEC-002).

Owner: Ayushi. Results keep model_version and params_hash, and the job keeps rules_version,
so every stored answer can be traced to the versions that produced it.
"""
from backend.core.config import Settings
from backend.db.models import FINAL_STATUSES, TIERS
from backend.db.repositories import jobs, task_state
from backend.workers.agg.tasks import finalise_in_session

ACTIVE = ("queued", "running")


class JobAccessError(Exception):
    """The caller may not see this job (403), it does not exist (404) or cannot be cancelled (409)."""

    def __init__(self, code, message, status):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


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
    """Return the status body. Raises JobAccessError: 403 for someone else's job, 404 if unknown."""
    job = jobs.get_job_owned(session, job_id, user_id)
    if job is None:
        if jobs.job_exists(session, job_id):
            raise JobAccessError("FORBIDDEN", "You do not have access to this job.", 403)
        raise JobAccessError("JOB_NOT_FOUND", "Job not found.", 404)
    task_state.expire_overdue(session, job.id, Settings.from_env())
    tasks = jobs.get_tasks_owned(session, job.id, user_id)
    tasks.sort(key=lambda task: TIERS.index(task.tier) if task.tier in TIERS else len(TIERS))
    results = jobs.get_results_owned(session, [task.id for task in tasks], user_id)
    if tasks and all(task.status in FINAL_STATUSES for task in tasks):
        finalise_in_session(session, job.id)
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
