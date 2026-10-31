import logging
import os
import re
import subprocess
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.api.app import create_app
from backend.api.auth.sessions import SessionStore, issue_token
from backend.core.config import Settings
from backend.db.models import AuditLog, Base, Job, JobTask, Molecule, Result, User

ROOT = Path(__file__).resolve().parents[2]
SECRET = "unit-test-secret-unit-test-secret-1234"
KEPT_SMILES = "CCOC(=O)c1ccc(N)cc1"
BAD_SMILES = "CCCCCCCCN(CC)CC(C)(C)C1"
INJECTIONS = ["'; DROP TABLE users;--", "CC' OR '1'='1", "$(rm -rf /)", "`id`",
              "../../etc/passwd", "<script>alert(1)</script>"]


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(self.format(record))


class FakeRedis:
    def __init__(self):
        self.data = {}

    def set(self, key, value, ex=None):
        self.data[key] = value

    def expire(self, key, seconds):
        return key in self.data

    def delete(self, key):
        self.data.pop(key, None)


class SecurityApiTests(unittest.TestCase):  # SEC-002, SEC-004, SEC-007, OWASP A01 A03 A05 A09
    def setUp(self):
        self.handler = ListHandler()
        root = logging.getLogger()
        self.old_level = root.level
        root.addHandler(self.handler)
        root.setLevel(logging.DEBUG)
        self.addCleanup(self.restore_logging)

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
            owner = User(email="a@example.com", password_hash="x")
            other = User(email="b@example.com", password_hash="x")
            admin = User(email="c@example.com", password_hash="x", role="administrator")
            molecule = Molecule(inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N", canonical_smiles="CCO")
            session.add_all([owner, other, admin, molecule])
            session.flush()
            job = Job(molecule_id=molecule.id, user_id=owner.id, options={}, rules_version="0.1")
            session.add(job)
            session.flush()
            ml = JobTask(job_id=job.id, tier="ml", status="done")
            dl = JobTask(job_id=job.id, tier="dl", status="running")
            qml = JobTask(job_id=job.id, tier="qml", status="queued")
            session.add_all([ml, dl, qml])
            session.flush()
            session.add(Result(job_task_id=ml.id, model_version="1.0", params_hash="abc", payload={"x": 1}))
            session.commit()
            self.job_id = job.id
            self.users = {"owner": (owner.id, "researcher"), "other": (other.id, "researcher"),
                          "admin": (admin.id, "administrator")}

        env = patch.dict(os.environ, {"JWT_SECRET": SECRET})
        env.start()
        self.addCleanup(env.stop)
        self.sessions = SessionStore(FakeRedis(), 1800)
        self.celery = MagicMock()
        patches = (
            ("backend.api.molecules.submit_routes.session_scope", scope),
            ("backend.api.molecules.submit_routes.enqueue_task", MagicMock()),
            ("backend.api.molecules.submit_routes.load_targets",
             lambda settings=None: {"targets": [{"id": "EGFR"}]}),
            ("backend.api.jobs.routes.session_scope", scope),
            ("backend.api.jobs.routes.celery_app", self.celery),
            ("backend.api.auth.sessions.get_session_store", lambda: self.sessions),
        )
        for target, value in patches:
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.app = create_app({"TESTING": True})
        self.client = self.app.test_client()
        settings = Settings.from_env()
        self.tokens = {name: issue_token(self.sessions, SimpleNamespace(id=uid, role=role), settings)
                       for name, (uid, role) in self.users.items()}

    def restore_logging(self):
        root = logging.getLogger()
        root.removeHandler(self.handler)
        root.setLevel(self.old_level)

    def auth(self, who="owner"):
        return {"Authorization": f"Bearer {self.tokens[who]}"}

    def post(self, path, payload, who="owner"):
        return self.client.post(path, json=payload, headers=self.auth(who))

    def test_sec002_cross_user_requests_get_403_and_change_nothing(self):
        url = f"/api/jobs/{self.job_id}"
        for who in ("other", "admin"):
            self.assertEqual(self.client.get(url, headers=self.auth(who)).status_code, 403)
            self.assertEqual(self.client.delete(url, headers=self.auth(who)).status_code, 403)
        with Session(self.engine) as session:
            statuses = sorted(task.status for task in session.execute(select(JobTask)).scalars())
            results = len(session.execute(select(Result)).scalars().all())
            job = session.get(Job, self.job_id)
        self.assertEqual(statuses, ["done", "queued", "running"])
        self.assertEqual(results, 1)
        self.assertIsNone(job.cancelled_at)
        self.celery.control.revoke.assert_not_called()

    def test_sec004_injection_strings_are_rejected_safely(self):
        for payload in INJECTIONS:
            submit = self.post("/api/molecules/submit", {"smiles": payload})
            self.assertEqual(submit.status_code, 422, payload)
            validate = self.post("/api/molecules/validate", {"smiles": payload})
            self.assertLess(validate.status_code, 500, payload)
            for resp in (submit, validate):
                self.assertNotIn("Traceback", resp.get_data(as_text=True))
        resp = self.post("/api/molecules/submit", {"smiles": "CCO", "target_id": "EGFR' OR 1=1--"})
        self.assertEqual(resp.status_code, 422)
        with Session(self.engine) as session:
            self.assertEqual(len(session.execute(select(User)).scalars().all()), 3)
            self.assertEqual(len(session.execute(select(Job)).scalars().all()), 1)

    def test_sec004_oversized_body_is_413(self):
        big = b'{"smiles": "' + b"C" * (2 * 1024 * 1024) + b'"}'
        resp = self.client.post("/api/molecules/submit", data=big, content_type="application/json",
                                headers=self.auth())
        self.assertEqual(resp.status_code, 413)

    def test_sec004_wrong_types_are_rejected(self):
        for body in ({"smiles": 123}, {"smiles": ["CCO"]}, {"smiles": {"a": 1}},
                     {"smiles": "CCO", "tiers": "ml"}, {"smiles": "CCO", "target_id": ["EGFR"]}):
            self.assertEqual(self.post("/api/molecules/submit", body).status_code, 422, body)

    def test_sec007_no_structures_in_logs_or_audit_log(self):
        self.assertEqual(self.post("/api/molecules/submit", {"smiles": KEPT_SMILES}).status_code, 202)
        self.post("/api/molecules/validate", {"smiles": KEPT_SMILES})
        self.post("/api/molecules/submit", {"smiles": BAD_SMILES})
        self.post("/api/molecules/validate", {"smiles": BAD_SMILES})
        logs = "\n".join(self.handler.messages)
        with Session(self.engine) as session:
            audit = str([row.detail for row in session.execute(select(AuditLog)).scalars()])
        for structure in (KEPT_SMILES, BAD_SMILES):
            self.assertNotIn(structure, logs)
            self.assertNotIn(structure, audit)

    def test_owasp_a05_debug_off_and_cors_is_strict(self):
        self.assertFalse(self.app.debug)
        evil = self.client.get("/healthz", headers={"Origin": "https://evil.example"})
        self.assertNotIn("Access-Control-Allow-Origin", evil.headers)
        allowed = self.app.config["FRONTEND_ORIGIN"]
        good = self.client.get("/healthz", headers={"Origin": allowed})
        self.assertEqual(good.headers.get("Access-Control-Allow-Origin"), allowed)


def tracked_files():
    result = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise unittest.SkipTest("git is not available")
    return [line for line in result.stdout.splitlines() if line]


def read_text(path):
    file = ROOT / path
    if not file.is_file() or file.stat().st_size > 1_000_000:
        return ""
    return file.read_text(encoding="utf-8", errors="ignore")


class StaticChecks(unittest.TestCase):  # SEC-003, SEC-008, OWASP A02 A06 A08
    def test_sec003_env_file_is_not_tracked(self):
        self.assertNotIn(".env", tracked_files())

    def test_sec003_env_example_holds_no_secret_values(self):
        for line in read_text(".env.example").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, _, value = line.partition("=")
            if re.search(r"SECRET|PASSWORD|TOKEN|KEY", key.upper()):
                self.assertEqual(value.strip(), "", f"{key} must be empty in .env.example")

    def test_sec003_no_secret_patterns_in_source(self):
        patterns = [r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----", r"AKIA[0-9A-Z]{16}",
                    r"ghp_[A-Za-z0-9]{36}", r"(?i)(password|secret|api[_-]?key)\s*[:=]\s*['\"][^'\"\s$]{6,}['\"]"]
        roots = ("backend/", "deploy/", "frontend/src/", "training/", ".github/", "docker-compose.yml")
        suffixes = (".py", ".js", ".jsx", ".yml", ".yaml", ".json", ".conf", ".sh", ".toml", ".ini")
        findings = []
        for name in tracked_files():
            if name.startswith(roots) and name.endswith(suffixes) and "package-lock" not in name:
                text = read_text(name)
                findings += [f"{name}: {p}" for p in patterns if re.search(p, text)]
        self.assertEqual(findings, [])

    def test_sec008_every_requirement_is_pinned(self):
        for line in read_text("requirements.txt").splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "-")):
                self.assertIn("==", line, f"unpinned requirement: {line}")

    def test_sec008_no_unsafe_calls_in_backend(self):
        unsafe = [r"\bpickle\.loads?\(", r"(?m)^\s*import pickle\b", r"\bjoblib\.load\(", r"\byaml\.load\(",
                  r"(?<![\w.])eval\(", r"(?<![\w.])exec\(", r"shell\s*=\s*True", r"debug\s*=\s*True"]
        findings = []
        for name in tracked_files():
            if name.startswith("backend/") and name.endswith(".py"):
                text = read_text(name)
                findings += [f"{name}: {p}" for p in unsafe if re.search(p, text)]
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
    