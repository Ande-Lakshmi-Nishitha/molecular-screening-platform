"""Shared setup for the end-to-end flow tests (MSP-40; QA-TST, NFR-005, BR-004, QA-REL).

A test submits a molecule through the real API, finds the task messages that were queued, runs
each tier's worker lifecycle (backend/workers/common/task_lifecycle.py) on them, and reads the
job back through the real status endpoint. Everything runs in one process: an in-memory
SQLite database, a fake Redis for sessions, and a mock queue that only records the messages.
No Docker, broker or model files are needed.

The tier models are small fakes with the same interface as the real predictors:
predict_smiles(smiles) returns an object with model_version, params_hash and payload.
"""
# pylint: disable=missing-function-docstring,missing-class-docstring,too-few-public-methods,unused-argument
import logging
import os
import unittest
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.api.app import create_app
from backend.api.auth.sessions import SessionStore, issue_token
from backend.core.config import Settings
from backend.db.models import Base, JobTask, Result, User
from backend.db.repositories import task_state
from backend.workers.common import task_lifecycle

SECRET = "unit-test-secret-unit-test-secret-1234"
ETHANOL = "CCO"  # 3 heavy atoms: eligible for ML and QML (FR-021)
LOG = logging.getLogger("msp.test.flow")


class FakeRedis:
    """Just enough Redis for the session store."""

    def __init__(self):
        self.data = {}

    def set(self, key, value, ex=None):
        self.data[key] = value

    def expire(self, key, seconds):
        return key in self.data

    def delete(self, key):
        self.data.pop(key, None)


class FakeMlModel:
    """Stands in for the ML predictor: a LogS value with its 90% interval."""

    def predict_smiles(self, smiles):  # pylint: disable=unused-argument
        return SimpleNamespace(model_version="ml-test-1.0", params_hash="ml-hash",
                               payload={"logS": -2.5, "logS_interval_90": [-3.1, -1.9]})


class StubQmlModel:
    """Stands in for the QML predictor. The result says it is a stub, never a real calculation."""

    def predict_smiles(self, smiles, should_cancel=None):  # pylint: disable=unused-argument
        return SimpleNamespace(model_version="stub-1.0", params_hash="stub-hash",
                               payload={"backend": "stub", "excitation_gap_hartree": 0.5,
                                        "label": "stub result, not a quantum calculation"})


class RaisingModel:
    """A predictor whose computation fails with the given exception."""

    def __init__(self, error):
        self.error = error

    def predict_smiles(self, smiles, should_cancel=None):  # pylint: disable=unused-argument
        raise self.error


def serving(model):
    """A get_model(target_id) that hands out this model."""
    return lambda _target: model


class FlowTestCase(unittest.TestCase):  # pylint: disable=too-many-instance-attributes
    """Real API and real worker lifecycle over an in-memory database."""

    def setUp(self):
        self.engine = create_engine("sqlite://", poolclass=StaticPool,
                                    connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        engine = self.engine

        @contextmanager
        def scope():
            session = Session(engine)
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        self.scope = scope
        with Session(engine) as session:
            user = User(email="flow@example.com", password_hash="x")
            session.add(user)
            session.commit()
            self.user_id = user.id

        env = patch.dict(os.environ, {"JWT_SECRET": SECRET})
        env.start()
        self.addCleanup(env.stop)
        self.sessions = SessionStore(FakeRedis(), 1800)
        self.enqueue = MagicMock()  # the queue: it only records which task ids were sent where
        for target, value in (
                ("backend.api.molecules.submit_routes.session_scope", scope),
                ("backend.api.molecules.submit_routes.enqueue_task", self.enqueue),
                ("backend.api.jobs.routes.session_scope", scope),
                ("backend.api.jobs.routes.celery_app", MagicMock()),
                ("backend.api.auth.sessions.get_session_store", lambda: self.sessions)):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = create_app({"TESTING": True}).test_client()
        token = issue_token(self.sessions, SimpleNamespace(id=self.user_id, role="researcher"),
                            Settings.from_env())
        self.headers = {"Authorization": f"Bearer {token}"}
        self.job_id = None
        self.task_ids = {}

    # ---- the user's side (API) ----
    def submit(self, smiles=ETHANOL, **extra):
        resp = self.client.post("/api/molecules/submit", json={"smiles": smiles, **extra},
                                headers=self.headers)
        self.assertEqual(resp.status_code, 202, resp.get_data(as_text=True))
        body = resp.get_json()
        self.job_id = body["job_id"]
        # UUID objects, because that is what the database columns expect
        self.task_ids = {task["tier"]: uuid.UUID(task["id"]) for task in body["tasks"]}
        return body

    def status(self):
        resp = self.client.get(f"/api/jobs/{self.job_id}", headers=self.headers)
        self.assertEqual(resp.status_code, 200, resp.get_data(as_text=True))
        body = resp.get_json()
        body["by_tier"] = {task["tier"]: task for task in body["tasks"]}
        return body

    # ---- the worker's side (queue message -> task lifecycle) ----
    def queued_task_id(self, tier):
        """The task id the API put on this tier's queue. Fails if nothing was queued for it."""
        for call in self.enqueue.call_args_list:
            if call.args[1] == tier:
                return str(call.args[0])
        return self.fail(f"no message was queued for tier {tier}")

    def run_worker(self, tier, get_model):
        """Deliver the tier's queued message to its worker. Returns (outcome, retry delay)."""
        return task_lifecycle.run_task(self.queued_task_id(tier), tier, get_model, LOG, self.scope)

    # ---- the database, read directly ----
    def task_row(self, tier):
        with Session(self.engine) as session:
            task = session.get(JobTask, self.task_ids[tier])
            return SimpleNamespace(status=task.status, error=task.error, attempts=task.attempts)

    def result_count(self, tier):
        with Session(self.engine) as session:
            return len(session.execute(
                select(Result).where(Result.job_task_id == self.task_ids[tier])).scalars().all())

    def claim_and_die(self, tier):
        """A worker claims the task and is killed: the task stays running and nothing is written."""
        with self.scope() as session:
            return task_state.claim_task(session, self.task_ids[tier])

    def make_overdue(self, tier, hours=24):
        """Pretend the task started long ago, so its hard time limit has passed."""
        started = datetime.now(timezone.utc) - timedelta(hours=hours)
        with Session(self.engine) as session:
            session.execute(update(JobTask).where(JobTask.id == self.task_ids[tier])
                            .values(started_at=started))
            session.commit()
