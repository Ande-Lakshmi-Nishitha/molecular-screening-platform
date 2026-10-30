"""Celery task on queue ml.

Owner: Nishitha. Requirements: FR-010, FR-011, FR-029, FR-058..060, SEC-002, SEC-007, NFR-001.

Task name and argument are fixed by backend/celery_app.py: `msp.ml.run`, one argument, the
job_task_id as a string. The message holds only that ID. The molecule and everything else
is read from the database, and the result is written back by this worker only.

Life of one task (all state changes are conditional updates, so a repeated or late
delivery cannot do harm):

  queued/running -> running   claim; anything else (done, cancelled, ...) is left alone
  compute                     featurise, LogS + interval, 12 assay signals
  running -> done             and the result row is written (upsert on the task ID);
                              if the task was cancelled meanwhile, 0 rows change and the
                              result is dropped
  running -> failed           with MODEL_UNAVAILABLE, TIMEOUT or COMPUTE_ERROR

Log lines hold the task ID and the exception class, never the structure (SEC-007).

Not here yet (they belong to Ayushi's backend/workers/common/base_task.py, FR-058..060):
retry with growing delay and the finalise(job_id) message to queue agg.
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional, Tuple

from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import worker_process_init
from sqlalchemy import select, update

from backend.celery_app import TASK_NAMES, celery_app
from backend.db.models import Job, JobTask, Molecule, Result
from backend.db.session import session_scope
from backend.registry.loader import ModelLoadError
from backend.workers.ml.predictor import MlPredictor, MlResult, get_predictor

log = logging.getLogger("msp.worker.ml")

TIER = "ml"
ERROR_MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
ERROR_TIMEOUT = "TIMEOUT"
ERROR_COMPUTE = "COMPUTE_ERROR"

OUTCOME_DONE = "done"
OUTCOME_FAILED = "failed"
OUTCOME_DROPPED = "dropped"    # finished after the task was cancelled: result discarded
OUTCOME_IGNORED = "ignored"    # not an ml task in queued/running state: nothing to do


def _now():
    return datetime.now(timezone.utc)


def _claim(session, task_id) -> Tuple[bool, Optional[str]]:
    """Move queued/running to running. Returns (claimed, stored SMILES of the molecule)."""
    changed = session.execute(
        update(JobTask)
        .where(JobTask.id == task_id, JobTask.tier == TIER, JobTask.status.in_(("queued", "running")))
        .values(status="running", started_at=_now(), attempts=JobTask.attempts + 1)
    ).rowcount
    if changed != 1:
        return False, None
    smiles = session.execute(
        select(Molecule.canonical_smiles)
        .join(Job, Job.molecule_id == Molecule.id)
        .join(JobTask, JobTask.job_id == Job.id)
        .where(JobTask.id == task_id)
    ).scalar_one_or_none()
    return True, smiles


def _finish(session, task_id, result: MlResult) -> bool:
    """running -> done plus the result row. False if the task is no longer running."""
    changed = session.execute(
        update(JobTask).where(JobTask.id == task_id, JobTask.status == "running")
        .values(status="done", finished_at=_now(), error=None)
    ).rowcount
    if changed != 1:
        return False
    row = session.execute(select(Result).where(Result.job_task_id == task_id)).scalar_one_or_none()
    if row is None:
        session.add(Result(job_task_id=task_id, model_version=result.model_version,
                           params_hash=result.params_hash, payload=result.payload))
    else:
        row.model_version, row.params_hash, row.payload = result.model_version, result.params_hash, result.payload
    return True


def _fail(session, task_id, code: str) -> None:
    """queued/running -> failed with an error code (a cancelled task stays cancelled)."""
    session.execute(
        update(JobTask).where(JobTask.id == task_id, JobTask.status.in_(("queued", "running")))
        .values(status="failed", error=code, finished_at=_now())
    )


def _compute(smiles: Optional[str], get_model: Callable[[], MlPredictor]) -> Tuple[Optional[MlResult], str]:
    """Run the prediction. Returns (result, "") or (None, error code)."""
    # pylint: disable=broad-exception-caught
    try:
        predictor = get_model()
    except SoftTimeLimitExceeded:
        return None, ERROR_TIMEOUT
    except (ModelLoadError, OSError) as exc:
        log.error("ml models unavailable: %s", type(exc).__name__)
        return None, ERROR_MODEL_UNAVAILABLE
    try:
        if not smiles:
            raise LookupError("molecule not found")
        return predictor.predict_smiles(smiles), ""
    except SoftTimeLimitExceeded:
        return None, ERROR_TIMEOUT
    except Exception as exc:
        log.error("ml compute failed: %s", type(exc).__name__)
        return None, ERROR_COMPUTE


def run_ml_task(job_task_id, get_model: Callable[[], MlPredictor] = get_predictor) -> str:
    """Run one ML task end to end. Returns one of the OUTCOME_* strings."""
    try:
        task_id = uuid.UUID(str(job_task_id))
    except ValueError:
        log.warning("ml task ignored: the task ID is not a UUID")
        return OUTCOME_IGNORED

    with session_scope() as session:
        claimed, smiles = _claim(session, task_id)
    if not claimed:
        return OUTCOME_IGNORED

    result, code = _compute(smiles, get_model)
    if result is None:
        with session_scope() as session:
            _fail(session, task_id, code)
        log.error("ml task %s failed: %s", task_id, code)
        return OUTCOME_FAILED

    with session_scope() as session:
        stored = _finish(session, task_id, result)
    if not stored:
        log.info("ml task %s finished after it stopped running; result dropped", task_id)
        return OUTCOME_DROPPED
    return OUTCOME_DONE


@celery_app.task(name=TASK_NAMES[TIER])
def ml_run(job_task_id):
    """Celery entry point for queue ml."""
    return run_ml_task(job_task_id)


@worker_process_init.connect
def _load_models_at_start(**_kwargs):
    """Load the models when the worker process starts, so the first task is not slow.

    A failure is only logged: each task then reports MODEL_UNAVAILABLE (no fallback model),
    and the models load as soon as the files are fixed.
    """
    try:
        get_predictor()
    except (ModelLoadError, OSError) as exc:
        log.error("ml models could not be loaded at start-up: %s", type(exc).__name__)
