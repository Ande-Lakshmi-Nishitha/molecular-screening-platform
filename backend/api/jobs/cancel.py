"""Cancel a job (FR-056, FR-057).

Owner: Ayushi. Unfinished tasks become cancelled in one UPDATE; finished results are left alone.
"""
from datetime import datetime, timezone

from sqlalchemy import update

from backend.api.jobs.status import JobAccessError
from backend.core import audit
from backend.db.models import JobTask
from backend.db.repositories import jobs

ACTIVE = ("queued", "running")


def cancel_job(session, job_id, user_id, now=None):
    """Cancel the caller's job. Returns the ids of the tasks that were still unfinished.

    403 for someone else's job, 404 if unknown, 409 if every task is already final.
    """
    now = now or datetime.now(timezone.utc)
    job = jobs.get_job_owned(session, job_id, user_id)
    if job is None:
        if jobs.job_exists(session, job_id):
            raise JobAccessError("FORBIDDEN", "You do not have access to this job.", 403)
        raise JobAccessError("JOB_NOT_FOUND", "Job not found.", 404)
    tasks = jobs.get_tasks_owned(session, job.id, user_id)
    active_ids = [task.id for task in tasks if task.status in ACTIVE]
    if not active_ids:
        raise JobAccessError("JOB_NOT_CANCELLABLE", "Every task of this job has already finished.", 409)
    job.cancelled_at = now
    audit.record(session, "job_cancelled", user_id, {"job_id": str(job.id)})
    session.flush()
    session.execute(update(JobTask).where(JobTask.job_id == job.id, JobTask.status.in_(ACTIVE))
                    .values(status="cancelled", finished_at=now)
                    .execution_options(synchronize_session=False))
    session.expire_all()
    return active_ids
