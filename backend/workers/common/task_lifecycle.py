"""What a model-tier worker (ML, DL, QML) does with one task, on top of base_task (MSP-23, MSP-26).

Owner: Nishitha. Requirements: FR-029, FR-058..060, SEC-002, SEC-007, NFR-001.

The message holds only the job_task_id. The molecule and the target are read from the database,
and the result is written back by the worker only.

The state machine itself is Ayushi's and is not repeated here: backend/workers/common/base_task.py
(claim, compute, finish or fail, retry with a growing delay) over backend/db/repositories/task_state.py
(one conditional UPDATE per change, so a repeated, late or cancelled delivery cannot do harm).
This module adds what is specific to the model tiers:

  * the tier check: a task of another tier that reaches this queue is left alone
  * reading the stored SMILES and the job's target id after the claim
  * the model step: get the tier's model, predict, and say why it failed
      MODEL_UNAVAILABLE   the model is missing, refused or does not exist for the target (no fallback)
      TIMEOUT             the soft time limit (decided by base_task)
      COMPUTE_ERROR       anything else (decided by base_task)
  * retry only for short faults base_task calls transient; the entry point asks Celery to retry

Log lines hold the task id and the exception class, never the structure (SEC-007).
Pure SQLAlchemy and Celery code: no RDKit, torch or XGBoost imports.
"""
import uuid
from typing import Any, Callable, Optional, Tuple

from sqlalchemy import select

from backend.core.config import Settings
from backend.db.models import Job, JobTask, Molecule
from backend.registry.loader import ModelLoadError
from backend.workers.common import base_task

ERROR_MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
ERROR_TIMEOUT = "TIMEOUT"
ERROR_COMPUTE = "COMPUTE_ERROR"

OUTCOME_DONE = "done"
OUTCOME_FAILED = "failed"
OUTCOME_RETRY = "retry"        # a short fault: the entry point asks Celery to run it again after the delay
OUTCOME_DROPPED = "dropped"    # finished after the task was cancelled: result discarded
OUTCOME_IGNORED = "ignored"    # not a task of this tier in queued/running state: nothing to do


class TaskFailure(Exception):
    """A failure with a fixed reason. base_task.decide_failure stores `reason` as the task error."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def read_inputs(session, task_id) -> Tuple[Optional[str], Optional[str]]:
    """The stored SMILES and the job's target id for a task (None, None if the rows are gone)."""
    row = session.execute(
        select(Molecule.canonical_smiles, Job.target_id)
        .join(Job, Job.molecule_id == Molecule.id)
        .join(JobTask, JobTask.job_id == Job.id)
        .where(JobTask.id == task_id)
    ).first()
    return (row[0], row[1]) if row else (None, None)


def _tier_of(session, task_id) -> Optional[str]:
    return session.execute(select(JobTask.tier).where(JobTask.id == task_id)).scalar_one_or_none()


def _model_step(task_id, tier, get_model, log, scope) -> dict:
    """The compute() that base_task.run_task calls after the claim. Raises on failure."""
    # pylint: disable=broad-exception-caught
    with scope() as session:
        smiles, target_id = read_inputs(session, task_id)
    try:
        model = get_model(target_id)
    except (ModelLoadError, OSError) as exc:
        log.error("%s models unavailable: %s", tier, type(exc).__name__)
        raise TaskFailure(ERROR_MODEL_UNAVAILABLE) from exc
    try:
        if not smiles:
            raise LookupError("molecule not found")
        result = model.predict_smiles(smiles)
    except Exception as exc:
        log.error("%s compute failed: %s", tier, type(exc).__name__)
        raise
    return {"model_version": result.model_version, "params_hash": result.params_hash,
            "payload": result.payload}


def run_task(job_task_id, tier: str, get_model: Callable[[Optional[str]], Any], log, scope,
             max_retries: Optional[int] = None) -> Tuple[str, int]:
    """Run one task end to end. Returns (OUTCOME_*, retry delay in seconds; 0 unless retrying).

    get_model(target_id) returns an object with predict_smiles(smiles) -> an object with
    model_version, params_hash and payload. `scope` is the session context manager.
    """
    try:
        task_id = uuid.UUID(str(job_task_id))
    except ValueError:
        log.warning("%s task ignored: the task ID is not a UUID", tier)
        return OUTCOME_IGNORED, 0
    with scope() as session:
        if _tier_of(session, task_id) != tier:
            return OUTCOME_IGNORED, 0

    ran = []

    def compute():
        ran.append(True)
        return _model_step(task_id, tier, get_model, log, scope)

    if max_retries is None:
        max_retries = Settings.from_env().task_max_retries
    outcome, delay = base_task.run_task(task_id, compute, scope, max_retries)
    if outcome == "dropped" and not ran:
        return OUTCOME_IGNORED, 0   # never claimed: finished, cancelled or already running elsewhere
    if outcome == "dropped":
        log.info("%s task %s finished after it stopped running; result dropped", tier, task_id)
    elif outcome == "failed":
        log.error("%s task %s failed", tier, task_id)
    elif outcome == "retry":
        log.warning("%s task %s will be retried in %s s", tier, task_id, delay)
    return outcome, delay


def retry_or_return(task, outcome: str, delay: int) -> str:
    """For a bound Celery task: ask Celery to run it again after `delay` s, or return the outcome.

    base_task counts the attempts and stops retrying after the configured number, so the Celery
    limit is set one higher and never decides on its own.
    """
    if outcome == OUTCOME_RETRY:
        raise task.retry(countdown=delay, max_retries=Settings.from_env().task_max_retries + 1)
    return outcome
