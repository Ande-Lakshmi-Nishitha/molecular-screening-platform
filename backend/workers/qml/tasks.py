"""Celery task on queue qml (MSP-27).

Owner: Nishitha. Requirements: FR-029, FR-058..060, SEC-002, SEC-007, NFR-003, NFR-013.

Task name and argument are fixed by backend/celery_app.py: `msp.qml.run`, one argument, the
job_task_id as a string. The message holds only that ID. The molecule is read from the database
and the result is written back by this worker only.

The life of the task is Ayushi's backend/workers/common/base_task.py (claim, store, fail, retry
with a growing delay); backend/workers/common/task_lifecycle.py adds the model-tier part shared
with the ML and DL workers. This module supplies the QML part: the quantum backend
(backend/workers/qml/predictor.py). Concurrency is 1 by default (QML_CONCURRENCY), set on the
worker. A cancelled task stays cancelled and a late result is dropped; a real backend (MSP-28)
also stops between iterations through the should_cancel callback of estimate_gap().
"""
import logging

from backend.celery_app import TASK_NAMES, celery_app
from backend.db.session import session_scope
from backend.workers.common import task_lifecycle
from backend.workers.common.task_lifecycle import (  # pylint: disable=unused-import  (re-exported)
    ERROR_COMPUTE, ERROR_MODEL_UNAVAILABLE, ERROR_TIMEOUT, OUTCOME_DONE, OUTCOME_DROPPED,
    OUTCOME_FAILED, OUTCOME_IGNORED, OUTCOME_RETRY)
from backend.workers.qml.predictor import get_predictor

log = logging.getLogger("msp.worker.qml")

TIER = "qml"


def run_qml_task_with_delay(job_task_id, get_model=get_predictor):
    """Run one QML task end to end. get_model(target_id) returns the predictor.

    Returns (OUTCOME_* string, retry delay in seconds).
    """
    return task_lifecycle.run_task(job_task_id, TIER, get_model, log, session_scope)


def run_qml_task(job_task_id, get_model=get_predictor) -> str:
    """Run one QML task end to end. Returns one of the OUTCOME_* strings."""
    return run_qml_task_with_delay(job_task_id, get_model)[0]


@celery_app.task(bind=True, name=TASK_NAMES[TIER])
def qml_run(self, job_task_id):
    """Celery entry point for queue qml: runs the task, or asks Celery to retry a short fault."""
    outcome, delay = run_qml_task_with_delay(job_task_id)
    return task_lifecycle.retry_or_return(self, outcome, delay)
