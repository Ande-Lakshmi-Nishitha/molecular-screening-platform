import os
import unittest
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import create_engine, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.api.app import create_app
from backend.api.auth.sessions import SessionStore, issue_token
from backend.core.config import Settings
from backend.db.models import AuditLog, Base, Job, JobTask, Molecule, Result, User
from backend.db.repositories import task_state
from backend.workers.common import base_task

SECRET = "unit-test-secret-unit-test-secret-1234"


class FakeRedis:
    def __init__(self):
        self.data = {}

    def set(self, key, value, ex=None):
        self.data[key] = value

    def expire(self, key, seconds):
        return key in self.data

    def delete(self, key):
        self.data.pop(key, None)


class DecisionTests(unittest.TestCase):  # FR-058, FR-059
    def test_retry_delay_grows_and_is_capped(self):
        self.assertEqual([base_task.retry_delay(n) for n in (1, 2, 3)], [5, 10, 20])
        self.assertEqual(base_task.retry_delay(20), 300)

    def test_decide_failure(self):
        db_error = OperationalError("stmt", {}, Exception("db lost"))
        self.assertEqual(base_task.decide_failure(SoftTimeLimitExceeded(), 1, 3), ("fail", "TIMEOUT", 0))
        self.assertEqual(base_task.decide_failure(ConnectionError("x"), 1, 3), ("retry", None, 5))
        self.assertEqual(base_task.decide_failure(db_error, 3, 3), ("retry", None, 20))
        self.assertEqual(base_task.decide_failure(ConnectionError("x"), 4, 3), ("fail", "COMPUTE_ERROR", 0))
        self.assertEqual(base_task.decide_failure(ValueError("x"), 1, 3), ("fail", "COMPUTE_ERROR", 0))


class CancelAndRetryTests(unittest.TestCase):  # FR-056..FR-060, BR-004
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
            owner = User(email="a@example.com", password_hash="x")
            other = User(email="b@example.com", password_hash="x")
            molecule = Molecule(inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N", canonical_smiles="CCO")
            session.add_all([owner, other, molecule])
            session.flush()
            self.owner_id, self.other_id, self.molecule_id = owner.id, other.id, molecule.id
            session.commit()
        self.job_a, self.tasks_a = self.new_job({"ml": "done", "dl": "running", "qml": "queued"})
        self.job_b, _ = self.new_job({"ml": "done", "dl": "skipped", "qml": "failed"})

        env = patch.dict(os.environ, {"JWT_SECRET": SECRET})
        env.start()
        self.addCleanup(env.stop)
        self.sessions = SessionStore(FakeRedis(), 1800)
        self.celery = MagicMock()
        for target, value in (("backend.api.jobs.routes.session_scope", scope),
                              ("backend.api.jobs.routes.celery_app", self.celery),
                              ("backend.api.auth.sessions.get_session_store", lambda: self.sessions)):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = create_app({"TESTING": True}).test_client()
        settings = Settings.from_env()
        self.tokens = {
            "owner": issue_token(self.sessions, SimpleNamespace(id=self.owner_id, role="researcher"), settings),
            "other": issue_token(self.sessions, SimpleNamespace(id=self.other_id, role="researcher"), settings),
        }

    def new_job(self, statuses, started=None):
        started = started or {}
        with Session(self.engine) as session:
            job = Job(molecule_id=self.molecule_id, user_id=self.owner_id, options={}, rules_version="0.1")
            session.add(job)
            session.flush()
            ids = {}
            for tier, status in statuses.items():
                begin = started.get(tier, datetime.now(timezone.utc)) if status == "running" else None
                task = JobTask(job_id=job.id, tier=tier, status=status, started_at=begin)
                session.add(task)
                session.flush()
                ids[tier] = task.id
                if status == "done":
                    session.add(Result(job_task_id=task.id, model_version="1.0", params_hash="abc",
                                       payload={"x": 1}))
            session.commit()
            return job.id, ids

    def task_row(self, task_id):
        with Session(self.engine) as session:
            task = session.get(JobTask, task_id)
            return task.status, task.error, task.attempts

    def count_results(self, task_id):
        with Session(self.engine) as session:
            return len(session.execute(select(Result).where(Result.job_task_id == task_id)).scalars().all())

    def auth(self, who="owner"):
        return {"Authorization": f"Bearer {self.tokens[who]}"}

    def delete(self, job_id, who="owner"):
        return self.client.delete(f"/api/jobs/{job_id}", headers=self.auth(who))

    def execute(self, task_id, compute, max_retries=3):
        return base_task.run_task(task_id, compute, self.scope, max_retries)

    # cancel
    def test_cancel_stops_unfinished_tasks_and_keeps_results(self):
        resp = self.delete(self.job_a)
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        tasks = {t["tier"]: t for t in body["tasks"]}
        self.assertEqual({tier: t["status"] for tier, t in tasks.items()},
                         {"ml": "done", "dl": "cancelled", "qml": "cancelled"})
        self.assertIsNotNone(tasks["ml"]["result"])
        self.assertIsNotNone(body["cancelled_at"])
        revoked = {call.args[0] for call in self.celery.control.revoke.call_args_list}
        self.assertEqual(revoked, {str(self.tasks_a["dl"]), str(self.tasks_a["qml"])})
        with Session(self.engine) as session:
            events = [row.event_type for row in session.execute(select(AuditLog)).scalars()]
        self.assertIn("job_cancelled", events)

    def test_cancel_errors(self):
        resp = self.delete(self.job_a, who="other")
        self.assertEqual((resp.status_code, resp.get_json()["error"]["code"]), (403, "FORBIDDEN"))
        self.assertEqual(self.task_row(self.tasks_a["dl"])[0], "running")
        self.assertEqual(self.delete(uuid.uuid4()).status_code, 404)
        self.assertEqual(self.delete("not-a-uuid").status_code, 404)
        resp = self.delete(self.job_b)
        self.assertEqual((resp.status_code, resp.get_json()["error"]["code"]), (409, "JOB_NOT_CANCELLABLE"))
        self.celery.control.revoke.assert_not_called()

    def test_cancel_needs_a_token(self):
        self.assertEqual(self.client.delete(f"/api/jobs/{self.job_a}").status_code, 401)

    def test_late_worker_write_is_dropped_after_cancel(self):
        self.delete(self.job_a)
        with self.scope() as session:
            self.assertFalse(task_state.finish_task(session, self.tasks_a["dl"], "1.0", "h", {"late": True}))
            self.assertEqual(task_state.claim_task(session, self.tasks_a["dl"]), 0)
            self.assertTrue(task_state.is_cancelled(session, self.tasks_a["dl"]))
        self.assertEqual(self.count_results(self.tasks_a["dl"]), 0)

    # redelivery and upsert
    def test_claim_is_safe_to_repeat_and_results_are_upserted(self):
        _, ids = self.new_job({"ml": "queued"})
        task = ids["ml"]
        with self.scope() as session:
            self.assertEqual(task_state.claim_task(session, task), 1)
        with self.scope() as session:
            self.assertEqual(task_state.claim_task(session, task), 2)  # redelivered message
        with self.scope() as session:
            self.assertTrue(task_state.finish_task(session, task, "1.0", "h1", {"v": 1}))
        with self.scope() as session:
            self.assertFalse(task_state.finish_task(session, task, "1.0", "h1", {"v": 9}))  # already done
        with Session(self.engine) as session:
            session.execute(update(JobTask).where(JobTask.id == task).values(status="running"))
            session.commit()
        with self.scope() as session:
            self.assertTrue(task_state.finish_task(session, task, "1.0", "h2", {"v": 2}))
        with Session(self.engine) as session:
            rows = session.execute(select(Result).where(Result.job_task_id == task)).scalars().all()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].payload, {"v": 2})

    # run_task
    def test_run_task_done(self):
        _, ids = self.new_job({"ml": "queued"})
        out = self.execute(ids["ml"], lambda: {"model_version": "1.0", "params_hash": "h", "payload": {"ok": True}})
        self.assertEqual(out, ("done", 0))
        self.assertEqual(self.task_row(ids["ml"])[0], "done")
        self.assertEqual(self.count_results(ids["ml"]), 1)

    def test_transient_errors_retry_with_growing_delay_then_fail(self):
        _, ids = self.new_job({"ml": "queued"})

        def boom():
            raise ConnectionError("database connection lost")

        outcomes = [self.execute(ids["ml"], boom) for _ in range(4)]
        self.assertEqual(outcomes, [("retry", 5), ("retry", 10), ("retry", 20), ("failed", 0)])
        self.assertEqual(self.task_row(ids["ml"]), ("failed", "COMPUTE_ERROR", 4))

    def test_soft_time_limit_fails_with_timeout(self):
        _, ids = self.new_job({"ml": "queued"})

        def slow():
            raise SoftTimeLimitExceeded()

        self.assertEqual(self.execute(ids["ml"], slow), ("failed", 0))
        self.assertEqual(self.task_row(ids["ml"])[:2], ("failed", "TIMEOUT"))

    def test_cancelled_task_is_not_computed(self):
        _, ids = self.new_job({"ml": "cancelled"})
        compute = MagicMock()
        self.assertEqual(self.execute(ids["ml"], compute), ("dropped", 0))
        compute.assert_not_called()

    # hard limit
    def test_overdue_running_task_is_failed_with_timeout_on_status_read(self):
        old = datetime.now(timezone.utc) - timedelta(hours=1)
        job, _ = self.new_job({"ml": "running", "dl": "running"},
                              started={"ml": old, "dl": datetime.now(timezone.utc)})
        body = self.client.get(f"/api/jobs/{job}", headers=self.auth()).get_json()
        tasks = {t["tier"]: t for t in body["tasks"]}
        self.assertEqual((tasks["ml"]["status"], tasks["ml"]["error"]), ("failed", "TIMEOUT"))
        self.assertEqual(tasks["dl"]["status"], "running")
        self.assertEqual(body["status"], "running")


if __name__ == "__main__":
    unittest.main()
    