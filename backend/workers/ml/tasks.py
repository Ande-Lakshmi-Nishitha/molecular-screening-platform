"""Celery task on queue ml.

Owner: Nishitha. Requirements: FR-010, FR-011, FR-029, FR-058..060, SEC-002, SEC-007, NFR-001.

Task name and argument are fixed by backend/celery_app.py: `msp.ml.run`, one argument, the
job_task_id as a string. The message holds only that ID. The molecule and everything else
is read from the database, and the result is written back by this worker only.

The life of the task is Ayushi's backend/workers/common/base_task.py (claim, store, fail, retry
with a growing delay); backend/workers/common/task_lifecycle.py adds the model-tier part shared
with the DL worker. This module supplies the ML part: featurise, LogS + interval, 12 assay signals.
"""
import logging

from celery.signals import worker_process_init

from backend.celery_app import TASK_NAMES, celery_app
from backend.db.session import session_scope
from backend.registry.loader import ModelLoadError
from backend.workers.common import task_lifecycle
from backend.workers.common.task_lifecycle import (  # pylint: disable=unused-import  (re-exported)
    ERROR_COMPUTE, ERROR_MODEL_UNAVAILABLE, ERROR_TIMEOUT, OUTCOME_DONE, OUTCOME_DROPPED,
    OUTCOME_FAILED, OUTCOME_IGNORED, OUTCOME_RETRY)
from backend.workers.ml.predictor import get_predictor

log = logging.getLogger("msp.worker.ml")

TIER = "ml"


def run_ml_task_with_delay(job_task_id, get_model=get_predictor):
    """Run one ML task end to end. Returns (OUTCOME_* string, retry delay in seconds)."""
    return task_lifecycle.run_task(job_task_id, TIER, lambda _target: get_model(), log, session_scope)


def run_ml_task(job_task_id, get_model=get_predictor) -> str:
    """Run one ML task end to end. Returns one of the OUTCOME_* strings."""
    return run_ml_task_with_delay(job_task_id, get_model)[0]


@celery_app.task(bind=True, name=TASK_NAMES[TIER])
def ml_run(self, job_task_id):
    """Celery entry point for queue ml: runs the task, or asks Celery to retry a short fault."""
    outcome, delay = run_ml_task_with_delay(job_task_id)
    return task_lifecycle.retry_or_return(self, outcome, delay)


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
