"""Paged job history for one user (MSP-37; FR-049, SEC-002).

Owner: Abheetha. Kept in its own file so Ayushi's jobs.py is not touched.
Every query filters on the caller's user id, as the other job queries do.
"""
from sqlalchemy import func, select

from backend.db.models import Job, JobTask, Molecule


def list_jobs_owned(session, user_id, limit, offset):
    """Return (total, rows) for this user's jobs, newest first.

    Each row is (job, molecule, tasks). total counts only this user's jobs.
    """
    total = session.execute(select(func.count()).select_from(Job).where(Job.user_id == user_id)).scalar_one()
    query = (select(Job, Molecule)
             .join(Molecule, Molecule.id == Job.molecule_id)
             .where(Job.user_id == user_id)
             .order_by(Job.created_at.desc(), Job.id.desc())
             .limit(limit).offset(offset))
    page = session.execute(query).all()
    tasks_by_job = {}
    job_ids = [job.id for job, _ in page]
    if job_ids:
        task_query = (select(JobTask).join(Job, Job.id == JobTask.job_id)
                      .where(JobTask.job_id.in_(job_ids), Job.user_id == user_id))
        for task in session.execute(task_query).scalars():
            tasks_by_job.setdefault(task.job_id, []).append(task)
    return total, [(job, molecule, tasks_by_job.get(job.id, [])) for job, molecule in page]
