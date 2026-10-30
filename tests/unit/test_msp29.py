import os
import time
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from flask import Flask
from kombu.exceptions import OperationalError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.api.auth.sessions import SessionStore, issue_token
from backend.api.molecules.submit_routes import submit_bp
from backend.core.config import Settings
from backend.db.models import AuditLog, Base, Job, JobTask, Molecule, User
from backend.services.molecule import parse_process
from backend.services.molecule.validate import validate_smiles

SECRET = "unit-test-secret-unit-test-secret-1234"
ETHANOL = "CCO"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def slow_compute(*_args):
    """Stands in for compute_validation to force the time limit (module level for spawn)."""
    time.sleep(60)
    return {}


class FakeRedis:
    def __init__(self):
        self.data = {}

    def set(self, key, value, ex=None):
        self.data[key] = value

    def expire(self, key, seconds):
        return key in self.data

    def delete(self, key):
        self.data.pop(key, None)


class SubmitTests(unittest.TestCase):  # FR-029, FR-030, FR-035, SEC-002
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

        with Session(engine) as session:
            user = User(email="r@example.com", password_hash="x")
            session.add(user)
            session.commit()
            self.user_id = user.id

        env = patch.dict(os.environ, {"JWT_SECRET": SECRET})
        env.start()
        self.addCleanup(env.stop)
        self.sessions = SessionStore(FakeRedis(), 1800)
        self.enqueue = MagicMock()
        patches = (
            ("backend.api.molecules.submit_routes.session_scope", scope),
            ("backend.api.molecules.submit_routes.enqueue_task", self.enqueue),
            ("backend.api.molecules.submit_routes.load_targets",
             lambda settings=None: {"targets": [{"id": "EGFR"}]}),
            ("backend.api.auth.sessions.get_session_store", lambda: self.sessions),
        )
        for target, value in patches:
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        app = Flask(__name__)
        app.register_blueprint(submit_bp)
        self.client = app.test_client()
        self.token = issue_token(self.sessions, SimpleNamespace(id=self.user_id, role="researcher"),
                                 Settings.from_env())

    def submit(self, body, token="default"):
        token = self.token if token == "default" else token
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return self.client.post("/api/molecules/submit", json=body, headers=headers)

    def count(self, model):
        with Session(self.engine) as session:
            return len(session.execute(select(model)).scalars().all())

    def test_valid_submit_creates_job_tasks_and_queues_eligible_tiers(self):
        resp = self.submit({"smiles": ETHANOL})
        self.assertEqual(resp.status_code, 202)
        body = resp.get_json()
        tasks = {t["tier"]: t for t in body["tasks"]}
        self.assertEqual(set(tasks), {"ml", "dl", "qml"})
        self.assertEqual(tasks["ml"]["status"], "queued")
        self.assertEqual((tasks["dl"]["status"], tasks["dl"]["skip_reason"]),
                         ("skipped", "NO_TARGET_SELECTED"))
        queued = [t for t in body["tasks"] if t["status"] == "queued"]
        self.assertEqual(self.enqueue.call_count, len(queued))
        self.assertEqual(self.count(Job), 1)
        self.assertEqual(self.count(JobTask), 3)

    def test_ineligible_tier_is_skipped_with_reason(self):
        tasks = {t["tier"]: t for t in self.submit({"smiles": ASPIRIN}).get_json()["tasks"]}
        self.assertEqual(tasks["qml"]["status"], "skipped")
        self.assertEqual(tasks["qml"]["skip_reason"], "NO_FRAGMENT")

    def test_job_belongs_to_the_logged_in_user_and_is_audited(self):
        self.submit({"smiles": ETHANOL})
        with Session(self.engine) as session:
            job = session.execute(select(Job)).scalar_one()
            self.assertEqual(job.user_id, self.user_id)
            events = [row.event_type for row in session.execute(select(AuditLog)).scalars()]
        self.assertIn("job_submitted", events)

    def test_unauthenticated_request_is_refused(self):
        resp = self.submit({"smiles": ETHANOL}, token=None)
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(self.count(Job), 0)
        self.enqueue.assert_not_called()

    def test_bad_smiles_gives_422_and_creates_nothing(self):
        resp = self.submit({"smiles": "not-a-smiles("})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.get_json()["error"]["code"], "VALIDATION_ERROR")
        self.assertEqual(self.count(Job), 0)
        self.enqueue.assert_not_called()

    def test_oversized_smiles_is_413_by_config_limit(self):  # FR-053, SAD 9.1
        with patch.dict(os.environ, {"MAX_SMILES_LEN": "10"}):
            resp = self.submit({"smiles": "C" * 11})
        self.assertEqual(resp.status_code, 413)
        self.assertEqual(resp.get_json()["error"]["code"], "INPUT_TOO_LARGE")
        self.assertNotIn("C" * 11, resp.get_data(as_text=True))
        self.assertEqual(self.count(Job), 0)
        self.enqueue.assert_not_called()

    def test_smiles_at_the_limit_is_accepted(self):
        with patch.dict(os.environ, {"MAX_SMILES_LEN": "10"}):
            resp = self.submit({"smiles": "C" * 10})
        self.assertEqual(resp.status_code, 202)

    def test_slow_parse_gives_parse_timeout_and_creates_nothing(self):  # FR-053
        started = time.monotonic()
        with patch.dict(os.environ, {"PARSE_TIMEOUT_S": "1"}), \
                patch.object(parse_process, "compute_validation", slow_compute):
            resp = self.submit({"smiles": ETHANOL})
        self.assertLess(time.monotonic() - started, 15)
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(resp.get_json()["error"]["code"], "PARSE_TIMEOUT")
        self.assertEqual(self.count(Job), 0)
        self.enqueue.assert_not_called()

    def test_submit_and_validate_use_the_same_result(self):  # MSP-20 shared validation
        for smiles in (ETHANOL, ASPIRIN, "CC(=O)[O-].[Na+]"):
            with self.subTest(smiles=smiles):
                expected = validate_smiles(smiles).to_dict()
                body = self.submit({"smiles": smiles}).get_json()
                self.assertEqual(body["molecule"],
                                 {k: v for k, v in expected.items() if k != "tiers"})
                skips = {t["tier"]: t["skip_reason"] for t in body["tasks"]}
                self.assertEqual(skips, {tier: d["skip_reason"]
                                         for tier, d in expected["tiers"].items()})

    def test_missing_smiles_and_non_json_give_422(self):
        self.assertEqual(self.submit({}).status_code, 422)
        resp = self.client.post("/api/molecules/submit", data="x", content_type="text/plain",
                                headers={"Authorization": f"Bearer {self.token}"})
        self.assertEqual(resp.status_code, 422)

    def test_unknown_target_is_refused(self):
        resp = self.submit({"smiles": ETHANOL, "target_id": "NOPE"})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(self.count(Job), 0)

    def test_requested_unavailable_tier_is_rejected_with_reason(self):
        resp = self.submit({"smiles": ETHANOL, "tiers": ["dl"]})
        self.assertEqual(resp.status_code, 422)
        error = resp.get_json()["error"]
        self.assertEqual(error["code"], "TIER_UNAVAILABLE")
        self.assertEqual(error["details"]["tiers"]["dl"], "NO_TARGET_SELECTED")
        self.assertEqual(self.count(Job), 0)

    def test_only_requested_tiers_get_tasks(self):
        resp = self.submit({"smiles": ETHANOL, "tiers": ["ml"]})
        self.assertEqual([t["tier"] for t in resp.get_json()["tasks"]], ["ml"])

    def test_same_molecule_twice_reuses_one_molecule_row(self):
        self.submit({"smiles": ETHANOL})
        self.submit({"smiles": ETHANOL})
        self.assertEqual(self.count(Molecule), 1)
        self.assertEqual(self.count(Job), 2)

    def test_queue_down_gives_503_and_marks_tasks_failed(self):
        self.enqueue.side_effect = OperationalError("broker down")
        resp = self.submit({"smiles": ETHANOL})
        self.assertEqual(resp.status_code, 503)
        with Session(self.engine) as session:
            tasks = {t.tier: t for t in session.execute(select(JobTask)).scalars()}
        self.assertEqual(tasks["ml"].status, "failed")
        self.assertEqual(tasks["ml"].error, "ENQUEUE_FAILED")
        self.assertEqual(tasks["dl"].status, "skipped")


if __name__ == "__main__":
    unittest.main()
    