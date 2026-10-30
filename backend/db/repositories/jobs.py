"""Job queries: molecules, jobs and task rows (FR-035, FR-058..060, SEC-002).

Owner: Ayushi. Every read of jobs, tasks and results takes the caller's user id and filters on
it. Only workers write results. Workers add functions here only after agreeing them.
"""
from datetime import datetime, timezone

from sqlalchemy import select

from backend.db.models import Job, JobTask, Molecule, Result


def _find_molecule(session, inchikey):
    return session.execute(select(Molecule).where(Molecule.inchikey == inchikey)).scalar_one_or_none()


def get_or_create_molecule(session, std, source="user"):
    """Return the molecule row for this InChIKey, creating it if it is new."""
    molecule = _find_molecule(session, std.inchikey)
    if molecule is None:
        molecule = Molecule(inchikey=std.inchikey, canonical_smiles=std.canonical_smiles,
                            formula=std.formula, heavy_atoms=std.heavy_atoms, source=source)
        session.add(molecule)
        session.flush()
    return molecule


def create_job(session, molecule_id, user_id, target_id, options, rules_version):
    """Insert a job owned by this user and flush so the id is set."""
    job = Job(molecule_id=molecule_id, user_id=user_id, target_id=target_id,
              options=options, rules_version=rules_version)
    session.add(job)
    session.flush()
    return job


def create_tasks(session, job_id, decisions, now=None):
    """One task row per tier: queued if eligible, otherwise skipped with its reason."""
    now = now or datetime.now(timezone.utc)
    tasks = []
    for tier, decision in decisions.items():
        if decision.eligible:
            task = JobTask(job_id=job_id, tier=tier, status="queued")
        else:
            task = JobTask(job_id=job_id, tier=tier, status="skipped",
                           skip_reason=decision.skip_reason, finished_at=now)
        session.add(task)
        tasks.append(task)
    session.flush()
    return tasks


def mark_tasks_failed(session, task_ids, error):
    """Mark tasks failed (used when the queue is unreachable)."""
    if not task_ids:
        return
    now = datetime.now(timezone.utc)
    for task in session.execute(select(JobTask).where(JobTask.id.in_(task_ids))).scalars():
        task.status = "failed"
        task.error = error
        task.finished_at = now


def get_job_owned(session, job_id, user_id):
    """The job if it belongs to this user, otherwise None."""
    query = select(Job).where(Job.id == job_id, Job.user_id == user_id)
    return session.execute(query).scalar_one_or_none()


def job_exists(session, job_id):
    """True if any job has this id (used only to tell 403 from 404, never to return data)."""
    return session.execute(select(Job.id).where(Job.id == job_id)).first() is not None


def get_tasks_owned(session, job_id, user_id):
    """The tasks of this user's job."""
    query = (select(JobTask).join(Job, Job.id == JobTask.job_id)
             .where(JobTask.job_id == job_id, Job.user_id == user_id))
    return session.execute(query).scalars().all()


def get_results_owned(session, task_ids, user_id):
    """Results of this user's tasks, keyed by task id."""
    if not task_ids:
        return {}
    query = (select(Result).join(JobTask, JobTask.id == Result.job_task_id)
             .join(Job, Job.id == JobTask.job_id)
             .where(Result.job_task_id.in_(task_ids), Job.user_id == user_id))
    return {row.job_task_id: row for row in session.execute(query).scalars()}
