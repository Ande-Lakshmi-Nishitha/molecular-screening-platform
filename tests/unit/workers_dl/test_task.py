"""Unit tests for the DL worker task: lifecycle, targets, failures, labels (MSP-26).

Covers FR-015..FR-020, FR-029, FR-058..060, SEC-002, SEC-007, NFR-006. The database is an
in-memory SQLite one and the models are the tiny real files from dl_fixtures.py.
Skipped when torch is not installed (requirements-dl.txt).

Run from the repo root with:  python -m unittest discover -s tests -t .
"""
# pylint: disable=missing-function-docstring,missing-class-docstring,protected-access,import-outside-toplevel
import json
import logging
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.db.models import Base, Job, JobTask, Molecule, Result, User

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

if torch is not None:
    from celery.exceptions import SoftTimeLimitExceeded

    from backend.api.jobs.status import build_status
    from backend.celery_app import TASK_NAMES, celery_app
    from backend.registry.loader import ModelLoadError
    from backend.workers.dl import predictor, tasks
    from tests.unit.workers_dl import dl_fixtures as fx

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


@unittest.skipIf(torch is None, "torch is not installed")
class TaskCase(unittest.TestCase):
    quiet = logging.NullHandler()  # keeps expected error lines off the test output

    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # pylint: disable=consider-using-with
        cls.addClassCleanup(tmp.cleanup)
        cls.root = fx.copy_of_built(Path(tmp.name))

    def setUp(self):
        predictor.reset_predictor()
        self.addCleanup(predictor.reset_predictor)
        self.engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
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

        tasks.log.addHandler(self.quiet)
        self.addCleanup(tasks.log.removeHandler, self.quiet)
        patcher = patch.object(tasks, "session_scope", scope)
        patcher.start()
        self.addCleanup(patcher.stop)
        with Session(engine) as session:
            user = User(email="r@example.com", password_hash="x")
            session.add(user)
            session.flush()
            molecule = Molecule(inchikey="BSYNRYMUTXBXSQ-UHFFFAOYSA-N", canonical_smiles=ASPIRIN,
                                formula="C9H8O4", heavy_atoms=13)
            session.add(molecule)
            session.flush()
            session.commit()
            self.user_id, self.molecule_id = user.id, molecule.id
        self.dl_id, self.ml_id, self.job_id = self.new_job("TEST_TARGET")

    def new_job(self, target_id, smiles=None):
        with Session(self.engine) as session:
            molecule_id = self.molecule_id
            if smiles:
                molecule = Molecule(inchikey=uuid.uuid4().hex[:27].upper(), canonical_smiles=smiles)
                session.add(molecule)
                session.flush()
                molecule_id = molecule.id
            job = Job(molecule_id=molecule_id, user_id=self.user_id, target_id=target_id)
            session.add(job)
            session.flush()
            dl = JobTask(job_id=job.id, tier="dl", status="queued")
            ml = JobTask(job_id=job.id, tier="ml", status="queued")
            session.add_all([dl, ml])
            session.commit()
            return dl.id, ml.id, job.id

    def models(self, target_id=None):
        return predictor.get_predictor(target_id, fx.models_dir(self.root), self.root / "cards", fx.TARGETS)

    def run_task(self, task_id=None, model=None):
        get_model = model or self.models
        return tasks.run_dl_task(str(task_id or self.dl_id), get_model)

    def row(self, task_id=None):
        with Session(self.engine) as session:
            return session.get(JobTask, task_id or self.dl_id)

    def results(self):
        with Session(self.engine) as session:
            return session.execute(select(Result)).scalars().all()

    def set_status(self, status, task_id=None):
        with Session(self.engine) as session:
            session.get(JobTask, task_id or self.dl_id).status = status
            session.commit()

    def assert_failed(self, code, task_id=None):
        row = self.row(task_id)
        self.assertEqual((row.status, row.error), ("failed", code))
        self.assertEqual(self.results(), [])


class SuccessTests(TaskCase):
    def test_a_queued_task_becomes_done_with_a_labelled_result(self):
        self.assertEqual(self.run_task(), tasks.OUTCOME_DONE)
        row = self.row()
        self.assertEqual((row.status, row.error, row.attempts), ("done", None, 1))
        self.assertIsNotNone(row.started_at)
        self.assertIsNotNone(row.finished_at)
        result, = self.results()
        self.assertEqual(result.job_task_id, self.dl_id)
        self.assertEqual(result.model_version, "dl_test_target_gnn@1.0.0")
        self.assertRegex(result.params_hash, r"^[0-9a-f]{64}$")
        self.assertEqual(result.payload["tier"], "dl")
        self.assertEqual(result.payload["activity"]["unit"], "pIC50")
        self.assertEqual(result.payload["activity"]["n_models"], 5)

    def test_fr020_the_job_target_decides_which_models_run(self):
        other_dl, _, _ = self.new_job("OTHER_TARGET")
        self.assertEqual(self.run_task(), tasks.OUTCOME_DONE)
        self.assertEqual(self.run_task(other_dl), tasks.OUTCOME_DONE)
        by_task = {row.job_task_id: row for row in self.results()}
        self.assertEqual(by_task[self.dl_id].payload["target"]["id"], "TEST_TARGET")
        self.assertEqual(by_task[other_dl].payload["target"]["id"], "OTHER_TARGET")
        self.assertEqual(by_task[other_dl].payload["activity"]["unit"], "pKi")
        self.assertEqual(by_task[other_dl].model_version, "dl_other_target_gnn@1.0.0")

    def test_the_stored_result_is_what_the_status_endpoint_shows(self):
        self.run_task()
        with Session(self.engine) as session:
            body = build_status(session, self.job_id, self.user_id)
        dl = next(task for task in body["tasks"] if task["tier"] == "dl")
        self.assertEqual(dl["status"], "done")
        self.assertEqual(dl["result"]["model_version"], "dl_test_target_gnn@1.0.0")
        self.assertEqual(dl["result"]["payload"]["activity"]["unit"], "pIC50")
        self.assertEqual(body["target_id"], "TEST_TARGET")

    def test_a_redelivered_message_does_nothing(self):
        self.assertEqual(self.run_task(), tasks.OUTCOME_DONE)
        before = self.results()[0].payload
        self.assertEqual(self.run_task(), tasks.OUTCOME_IGNORED)
        self.assertEqual(len(self.results()), 1)
        self.assertEqual(self.results()[0].payload, before)
        self.assertEqual(self.row().attempts, 1)

    def test_a_task_found_running_after_a_crash_is_run_again_once_more(self):
        self.set_status("running")
        self.assertEqual(self.run_task(), tasks.OUTCOME_DONE)
        self.assertEqual(self.row().attempts, 1)

    def test_only_the_dl_task_is_touched(self):
        self.run_task()
        with Session(self.engine) as session:
            self.assertEqual(session.get(JobTask, self.ml_id).status, "queued")


class IgnoredTests(TaskCase):
    def test_tasks_that_are_not_queued_or_running_are_left_alone(self):
        for status in ("done", "failed", "skipped", "cancelled"):
            with self.subTest(status=status):
                self.set_status(status)
                self.assertEqual(self.run_task(), tasks.OUTCOME_IGNORED)
                self.assertEqual(self.row().status, status)
        self.assertEqual(self.results(), [])

    def test_an_ml_task_id_is_ignored_by_the_dl_worker(self):
        self.assertEqual(self.run_task(self.ml_id), tasks.OUTCOME_IGNORED)
        with Session(self.engine) as session:
            self.assertEqual(session.get(JobTask, self.ml_id).status, "queued")

    def test_unknown_and_malformed_ids_are_ignored(self):
        self.assertEqual(self.run_task(uuid.uuid4()), tasks.OUTCOME_IGNORED)
        self.assertEqual(tasks.run_dl_task("not-a-uuid", self.models), tasks.OUTCOME_IGNORED)
        self.assertEqual(tasks.run_dl_task(None, self.models), tasks.OUTCOME_IGNORED)

    def test_cancelled_while_computing_the_result_is_dropped(self):
        def cancel_then_give_model(target_id):
            self.set_status("cancelled")
            return self.models(target_id)
        self.assertEqual(self.run_task(model=cancel_then_give_model), tasks.OUTCOME_DROPPED)
        self.assertEqual(self.row().status, "cancelled")
        self.assertEqual(self.results(), [])

    def test_cancelled_while_computing_stays_cancelled_even_if_the_model_fails(self):
        def cancel_then_fail(_target):
            self.set_status("cancelled")
            raise ModelLoadError("x")
        self.assertEqual(self.run_task(model=cancel_then_fail), tasks.OUTCOME_FAILED)
        self.assertEqual(self.row().status, "cancelled")


class FailureTests(TaskCase):
    def test_a_job_without_a_target_fails_as_model_unavailable(self):
        task_id, _, _ = self.new_job(None)
        self.assertEqual(self.run_task(task_id), tasks.OUTCOME_FAILED)
        self.assert_failed(tasks.ERROR_MODEL_UNAVAILABLE, task_id)

    def test_a_target_that_is_not_configured_fails_as_model_unavailable(self):
        task_id, _, _ = self.new_job("NOT_A_TARGET")
        self.assertEqual(self.run_task(task_id), tasks.OUTCOME_FAILED)
        self.assert_failed(tasks.ERROR_MODEL_UNAVAILABLE, task_id)

    def test_missing_or_tampered_models_fail_as_model_unavailable_without_a_fallback(self):
        card = json.loads(fx.card_file(self.root, "dl_test_target_gnn").read_text(encoding="utf-8"))
        name = next(a["file"] for a in card["artifacts"] if a.get("role") == "gnn_member")
        path = fx.models_dir(self.root) / "dl" / name
        saved = path.read_bytes()
        path.write_bytes(saved + b"x")
        try:
            self.assertEqual(self.run_task(), tasks.OUTCOME_FAILED)
        finally:
            path.write_bytes(saved)
        self.assert_failed(tasks.ERROR_MODEL_UNAVAILABLE)

    def test_one_target_failing_does_not_stop_another(self):
        other_dl, _, _ = self.new_job("OTHER_TARGET")
        broken = fx.card_file(self.root, "dl_test_target_gnn")
        saved = broken.read_bytes()
        broken.write_bytes(b"{ broken")
        try:
            self.assertEqual(self.run_task(), tasks.OUTCOME_FAILED)
            self.assertEqual(self.run_task(other_dl), tasks.OUTCOME_DONE)
        finally:
            broken.write_bytes(saved)

    def test_an_os_error_while_loading_is_model_unavailable(self):
        def broken(_target):
            raise OSError("disk")
        self.assertEqual(self.run_task(model=broken), tasks.OUTCOME_FAILED)
        self.assert_failed(tasks.ERROR_MODEL_UNAVAILABLE)

    def test_soft_time_limit_while_loading_or_predicting_is_timeout(self):
        def slow_load(_target):
            raise SoftTimeLimitExceeded()
        self.assertEqual(self.run_task(model=slow_load), tasks.OUTCOME_FAILED)
        self.assert_failed(tasks.ERROR_TIMEOUT)
        self.set_status("queued")

        class Slow:
            @staticmethod
            def predict_smiles(_smiles):
                raise SoftTimeLimitExceeded()
        self.assertEqual(self.run_task(model=lambda _t: Slow()), tasks.OUTCOME_FAILED)
        self.assert_failed(tasks.ERROR_TIMEOUT)

    def test_a_structure_the_graph_cannot_represent_is_a_compute_error(self):
        task_id, _, _ = self.new_job("TEST_TARGET", smiles="C[Si](C)(C)C")
        self.assertEqual(self.run_task(task_id), tasks.OUTCOME_FAILED)
        self.assert_failed(tasks.ERROR_COMPUTE, task_id)

    def test_any_other_error_is_a_compute_error(self):
        class Broken:
            @staticmethod
            def predict_smiles(_smiles):
                raise RuntimeError("boom")
        self.assertEqual(self.run_task(model=lambda _t: Broken()), tasks.OUTCOME_FAILED)
        self.assert_failed(tasks.ERROR_COMPUTE)

    def test_sec007_log_lines_hold_the_task_id_and_exception_class_never_the_structure(self):
        class Leaky:
            @staticmethod
            def predict_smiles(smiles):
                raise ValueError(f"cannot handle {smiles}")
        with self.assertLogs("msp.worker.dl", level="INFO") as logs:
            self.run_task(model=lambda _t: Leaky())
        text = "\n".join(logs.output)
        self.assertIn(str(self.dl_id), text)
        self.assertIn("ValueError", text)
        self.assertNotIn(ASPIRIN, text)
        self.assertNotIn("cannot handle", text)

    def test_a_failed_task_runs_again_only_if_it_is_queued_again(self):
        self.run_task(model=lambda _t: (_ for _ in ()).throw(ModelLoadError("x")))
        self.assertEqual(self.run_task(), tasks.OUTCOME_IGNORED)   # failed is final here
        self.set_status("queued")                                   # a retry re-queues it
        self.assertEqual(self.run_task(), tasks.OUTCOME_DONE)


class CeleryWiringTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_the_task_is_registered_under_the_agreed_name(self):
        self.assertEqual(TASK_NAMES["dl"], "msp.dl.run")
        self.assertIn("msp.dl.run", celery_app.tasks)
        self.assertEqual(tasks.dl_run.name, "msp.dl.run")

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_the_entry_point_takes_one_id_and_returns_the_outcome(self):
        with patch.object(tasks, "run_dl_task_with_delay", return_value=("done", 0)) as run:
            self.assertEqual(tasks.dl_run.run("abc"), "done")
        run.assert_called_once_with("abc")

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_start_up_loads_every_configured_target_and_logs_failures_only(self):
        loaded = []

        def fake_get(target_id):
            loaded.append(target_id)
            if target_id == "BAD":
                raise ModelLoadError("no files")
        targets = {"targets": [{"id": "GOOD"}, {"id": "BAD"}, {"id": "ALSO_GOOD"}]}
        with patch.object(tasks, "load_targets", return_value=targets), patch.object(tasks, "get_predictor", fake_get):
            with self.assertLogs("msp.worker.dl", level="ERROR") as logs:
                tasks._load_models_at_start()
        self.assertEqual(loaded, ["GOOD", "BAD", "ALSO_GOOD"])
        self.assertIn("ModelLoadError", "\n".join(logs.output))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_start_up_survives_an_unreadable_target_file(self):
        with patch.object(tasks, "load_targets", side_effect=OSError("gone")):
            with self.assertLogs("msp.worker.dl", level="ERROR") as logs:
                tasks._load_models_at_start()
        self.assertIn("OSError", "\n".join(logs.output))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_start_up_with_no_targets_does_nothing(self):
        with patch.object(tasks, "load_targets", return_value=None), patch.object(tasks, "get_predictor") as get:
            tasks._load_models_at_start()
        get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
