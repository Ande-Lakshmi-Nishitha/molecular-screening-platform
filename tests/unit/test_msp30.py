import os
import unittest
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.api.auth.sessions import SessionStore, issue_token
from backend.api.jobs.routes import jobs_bp
from backend.api.jobs.status import overall_status
from backend.core.config import Settings
from backend.db.models import Base, Job, JobTask, Molecule, Result, User

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


class OverallStatusTests(unittest.TestCase):  # FR-031
    def test_roll_up(self):
        cases = [
            (["queued", "queued", "skipped"], False, "queued"),
            (["done", "running", "skipped"], False, "running"),
            (["done", "done", "skipped"], False, "complete"),
            (["done", "failed", "skipped"], False, "partial"),
            (["failed", "skipped", "skipped"], False, "failed"),
            (["cancelled", "skipped"], True, "cancelled"),
            (["done", "cancelled"], True, "partial"),
        ]
        for statuses, cancelled, expected in cases:
            self.assertEqual(overall_status(statuses, cancelled), expected, statuses)


class JobStatusTests(unittest.TestCase):  # FR-031, FR-032, FR-036, SEC-002
    def setUp(self):
        engine = create_engine("sqlite://", poolclass=StaticPool,
                               connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)

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

        with Session(engine) as session:
            owner = User(email="a@example.com", password_hash="x")
            other = User(email="b@example.com", password_hash="x")
            molecule = Molecule(inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N", canonical_smiles="CCO")
            session.add_all([owner, other, molecule])
            session.flush()
            job = Job(molecule_id=molecule.id, user_id=owner.id, options={}, rules_version="0.1")
            session.add(job)
            session.flush()
            ml = JobTask(job_id=job.id, tier="ml", status="done")
            dl = JobTask(job_id=job.id, tier="dl", status="running")
            qml = JobTask(job_id=job.id, tier="qml", status="skipped", skip_reason="NO_FRAGMENT")
            session.add_all([qml, dl, ml])
            session.flush()
            session.add(Result(job_task_id=ml.id, model_version="1.0", params_hash="abc123",
                               payload={"logS": -2.1}))
            session.commit()
            self.job_id, self.owner_id, self.other_id = job.id, owner.id, other.id

        env = patch.dict(os.environ, {"JWT_SECRET": SECRET})
        env.start()
        self.addCleanup(env.stop)
        self.sessions = SessionStore(FakeRedis(), 1800)
        for target, value in (("backend.api.jobs.routes.session_scope", scope),
                              ("backend.api.auth.sessions.get_session_store", lambda: self.sessions)):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        app = Flask(__name__)
        app.register_blueprint(jobs_bp)
        self.client = app.test_client()
        settings = Settings.from_env()
        self.owner_token = issue_token(self.sessions, SimpleNamespace(id=self.owner_id, role="researcher"), settings)
        self.other_token = issue_token(self.sessions, SimpleNamespace(id=self.other_id, role="researcher"), settings)

    def get(self, job_id, token="owner"):
        value = {"owner": self.owner_token, "other": self.other_token}.get(token)
        headers = {"Authorization": f"Bearer {value}"} if value else {}
        return self.client.get(f"/api/jobs/{job_id}", headers=headers)

    def test_owner_sees_status_and_tasks_in_tier_order(self):
        resp = self.get(self.job_id)
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["status"], "running")
        self.assertEqual([t["tier"] for t in body["tasks"]], ["ml", "dl", "qml"])
        self.assertEqual(body["tasks"][2]["skip_reason"], "NO_FRAGMENT")

    def test_partial_results_appear_only_for_finished_tasks(self):
        tasks = {t["tier"]: t for t in self.get(self.job_id).get_json()["tasks"]}
        self.assertEqual(tasks["ml"]["result"]["payload"], {"logS": -2.1})
        self.assertIsNone(tasks["dl"]["result"])
        self.assertIsNone(tasks["qml"]["result"])

    def test_versions_are_persisted_and_returned(self):
        body = self.get(self.job_id).get_json()
        self.assertEqual(body["rules_version"], "0.1")
        self.assertEqual(body["tasks"][0]["result"]["model_version"], "1.0")
        self.assertEqual(body["tasks"][0]["result"]["params_hash"], "abc123")

    def test_another_users_job_is_a_404(self):
        resp = self.get(self.job_id, token="other")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.get_json()["error"]["code"], "JOB_NOT_FOUND")

    def test_unknown_and_malformed_ids_are_404(self):
        self.assertEqual(self.get(uuid.uuid4()).status_code, 404)
        self.assertEqual(self.get("not-a-uuid").status_code, 404)

    def test_unauthenticated_request_is_401(self):
        self.assertEqual(self.get(self.job_id, token=None).status_code, 401)


if __name__ == "__main__":
    unittest.main()
    