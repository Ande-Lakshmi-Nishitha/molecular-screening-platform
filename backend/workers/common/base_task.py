"""Shared worker logic: claim, compute, finish, retry with growing delay (FR-058, FR-059).

Owner: Ayushi. A worker task calls run_task() and, if it returns ("retry", delay), asks Celery to
retry after that many seconds. Messages carry only the task id; everything else comes from the database.
"""
from celery.exceptions import SoftTimeLimitExceeded, TimeLimitExceeded
from sqlalchemy.exc import OperationalError

from backend.db.repositories import task_state

RETRY_BASE_S = 5
RETRY_CAP_S = 300


class TransientError(Exception):
    """A short fault that is worth retrying (for example a lost connection)."""


TRANSIENT_ERRORS = (TransientError, ConnectionError, TimeoutError, OperationalError)


def retry_delay(attempt, base=RETRY_BASE_S, cap=RETRY_CAP_S):
    """Growing delay in seconds: 5, 10, 20, ... capped (TBD-C12, provisional)."""
    return min(cap, base * 2 ** max(0, attempt - 1))


def decide_failure(exc, attempts, max_retries):
    """Return (action, reason, delay): timeouts fail with TIMEOUT, transient errors retry, the rest fail."""
    if isinstance(exc, (SoftTimeLimitExceeded, TimeLimitExceeded)):
        return "fail", "TIMEOUT", 0
    if isinstance(exc, TRANSIENT_ERRORS) and attempts <= max_retries:
        return "retry", None, retry_delay(attempts)
    return "fail", "COMPUTE_ERROR", 0


def run_task(task_id, compute, scope, max_retries):
    """Claim the task, run compute(), then finish or fail it.

    compute() returns {"model_version", "params_hash", "payload"}. Returns (outcome, delay) where
    outcome is done, retry, failed or dropped (cancelled, or already finished).
    """
    with scope() as session:
        attempts = task_state.claim_task(session, task_id)
    if not attempts:
        return "dropped", 0
    try:
        output = compute()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        action, reason, delay = decide_failure(exc, attempts, max_retries)
        with scope() as session:
            if action == "retry":
                task_state.requeue_task(session, task_id)
                return "retry", delay
            task_state.fail_task(session, task_id, reason)
        return "failed", 0
    with scope() as session:
        stored = task_state.finish_task(session, task_id, output["model_version"],
                                        output["params_hash"], output["payload"])
    return ("done" if stored else "dropped"), 0
