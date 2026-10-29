"""Job queries: molecules, jobs and task rows (FR-035, FR-058..060).

Owner: Ayushi. Workers add functions here only after agreeing them with Ayushi.
"""
from datetime import datetime, timezone

from sqlalchemy import select

from backend.db.models import Job, JobTask, Molecule


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


def get_job(session, job_id):
    """Return the job row or None."""
    return session.get(Job, job_id)
