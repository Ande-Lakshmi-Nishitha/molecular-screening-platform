"""Celery task on queue dl.

Owner: Nishitha. Requirements: FR-015..FR-020, FR-029, FR-058..060, SEC-002, SEC-007, NFR-002.

Task name and argument are fixed by backend/celery_app.py: `msp.dl.run`, one argument, the
job_task_id as a string. The message holds only that ID. The molecule and the target are read
from the database, and the result is written back by this worker only.

The life of the task is Ayushi's backend/workers/common/base_task.py (claim, store, fail, retry
with a growing delay); backend/workers/common/task_lifecycle.py adds the model-tier part shared
with the ML worker. This module supplies the DL part: the target's five-model ensemble (mean and
SD) and its baseline.
"""
import logging

from celery.signals import worker_process_init

from backend.celery_app import TASK_NAMES, celery_app
from backend.core.config import load_targets
from backend.db.session import session_scope
from backend.registry.loader import ModelLoadError
from backend.workers.common import task_lifecycle
from backend.workers.common.task_lifecycle import (  # pylint: disable=unused-import  (re-exported)
    ERROR_COMPUTE, ERROR_MODEL_UNAVAILABLE, ERROR_TIMEOUT, OUTCOME_DONE, OUTCOME_DROPPED,
    OUTCOME_FAILED, OUTCOME_IGNORED, OUTCOME_RETRY)
from backend.workers.dl.predictor import get_predictor

log = logging.getLogger("msp.worker.dl")

TIER = "dl"


def run_dl_task_with_delay(job_task_id, get_model=get_predictor):
    """Run one DL task end to end. get_model(target_id) returns the target's predictor.

    Returns (OUTCOME_* string, retry delay in seconds).
    """
    return task_lifecycle.run_task(job_task_id, TIER, get_model, log, session_scope)


def run_dl_task(job_task_id, get_model=get_predictor) -> str:
    """Run one DL task end to end. Returns one of the OUTCOME_* strings."""
    return run_dl_task_with_delay(job_task_id, get_model)[0]


@celery_app.task(bind=True, name=TASK_NAMES[TIER])
def dl_run(self, job_task_id):
    """Celery entry point for queue dl: runs the task, or asks Celery to retry a short fault."""
    outcome, delay = run_dl_task_with_delay(job_task_id)
    return task_lifecycle.retry_or_return(self, outcome, delay)


@worker_process_init.connect
def _load_models_at_start(**_kwargs):
    """Load every configured target's models when the worker process starts.

    A failure is only logged (never the target's molecules): each task then reports
    MODEL_UNAVAILABLE for that target (no fallback model), and the models load as soon as the
    files are fixed.
    """
    try:
        targets = (load_targets() or {}).get("targets") or []
    except (OSError, ValueError) as exc:
        log.error("dl targets could not be read at start-up: %s", type(exc).__name__)
        return
    for target in targets:
        try:
            get_predictor(target.get("id"))
        except (ModelLoadError, OSError) as exc:
            log.error("dl models for %s could not be loaded at start-up: %s", target.get("id"), type(exc).__name__)
