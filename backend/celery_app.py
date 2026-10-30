"""Celery app: queues ml, dl, qml, agg; JSON only, acks_late, prefetch 1.

Owner: Ayushi. Requirements: FR-029, FR-060, NFR-006, NFR-013.
Workers must register their tasks under these fixed names. Each task takes one argument,
the job_task_id as a string, and loads everything else from the database:
  msp.ml.run (queue ml), msp.dl.run (queue dl), msp.qml.run (queue qml)
The Celery task id is the job_task_id, so a cancel can revoke exactly that task. The Redis
visibility timeout is above the longest hard limit, so a message is only redelivered after a
worker that died has really lost it (FR-060).
"""
from celery import Celery
from kombu import Queue

from backend.core.config import Settings

QUEUE_NAMES = ("ml", "dl", "qml", "agg")
TASK_NAMES = {"ml": "msp.ml.run", "dl": "msp.dl.run", "qml": "msp.qml.run"}
VISIBILITY_MARGIN_S = 600


def longest_hard_limit(settings):
    """The longest hard time limit of any tier, in seconds."""
    return max(settings.ml_hard_limit_s, settings.dl_hard_limit_s, settings.qml_hard_limit_s)


def make_celery(settings=None):
    """Create the Celery app with the agreed safe settings."""
    settings = settings or Settings.from_env()
    instance = Celery("msp", broker=settings.redis_url, backend=settings.redis_url)
    instance.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_default_queue="ml",
        task_queues=tuple(Queue(name) for name in QUEUE_NAMES),
        broker_transport_options={"visibility_timeout": longest_hard_limit(settings) + VISIBILITY_MARGIN_S},
        broker_connection_retry_on_startup=True,
        timezone="UTC",
        enable_utc=True,
    )
    return instance


celery_app = make_celery()  # pylint: disable=invalid-name
app = celery_app  # the worker entrypoint looks for an attribute named `app`


def tier_limits(settings, tier):
    """Soft and hard time limits for a tier, read from config (TBD-C12)."""
    return {"soft_time_limit": getattr(settings, f"{tier}_soft_limit_s"),
            "time_limit": getattr(settings, f"{tier}_hard_limit_s")}


def enqueue_task(job_task_id, tier, settings=None, app=None):  # pylint: disable=redefined-outer-name
    """Send one tier task to its own queue with its time limits; the Celery id is the task row id."""
    settings = settings or Settings.from_env()
    app = app or celery_app
    return app.send_task(TASK_NAMES[tier], args=[str(job_task_id)], queue=tier,
                         task_id=str(job_task_id), **tier_limits(settings, tier))
