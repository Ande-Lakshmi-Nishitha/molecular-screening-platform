"""Unit tests for the ML worker task: lifecycle, idempotency, cancellation, failures (MSP-23).

Covers FR-010, FR-011, FR-029, FR-058..060, SEC-002, SEC-007, NFR-006. The database is an
in-memory SQLite one and the models are the tiny real files from ml_fixtures.py.

Run from the repo root with:  python -m unittest discover -s tests -t .
"""
# pylint: disable=missing-function-docstring,missing-class-docstring,protected-access
import json
import logging
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.celery_app import TASK_NAMES, celery_app
from backend.db.models import Base, Job, JobTask, Molecule, Result, User
from backend.registry.loader import ModelLoadError
from backend.workers.ml import predictor, tasks
from tests.unit.workers_ml import ml_fixtures as fx

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


class TaskCase(unittest.TestCase):
    quiet = logging.NullHandler()  # keeps expected error lines off the test output

    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # pylint: disable=consider-using-with
        cls.addClassCleanup(tmp.cleanup)
        cls.root = fx.copy_of_built(Path(tmp.name))
        cls.model = predictor.MlPredictor.load(fx.models_dir(cls.root), cards_dir=cls.root / "cards")

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
            job = Job(molecule_id=molecule.id, user_id=user.id)
            session.add(job)
            session.flush()
            self.ml = JobTask(job_id=job.id, tier="ml", status="queued")
            self.dl = JobTask(job_id=job.id, tier="dl", status="queued")
            session.add_all([self.ml, self.dl])
            session.commit()
            self.ml_id, self.dl_id, self.molecule_id = self.ml.id, self.dl.id, molecule.id

    def run_task(self, task_id=None, model=None):
        get_model = model if callable(model) else (lambda: model or self.model)
        return tasks.run_ml_task(str(task_id or self.ml_id), get_model)

    def row(self, task_id=None):
        with Session(self.engine) as session:
            return session.get(JobTask, task_id or self.ml_id)

    def results(self):
        with Session(self.engine) as session:
            return session.execute(select(Result)).scalars().all()

    def set_status(self, status, task_id=None):
        with Session(self.engine) as session:
            session.get(JobTask, task_id or self.ml_id).status = status
            session.commit()


class SuccessTests(TaskCase):
    def test_fr010_fr011_a_queued_task_finishes_with_a_stored_labelled_result(self):
        self.assertEqual(self.run_task(), tasks.OUTCOME_DONE)
        row = self.row()
        self.assertEqual(row.status, "done")
        self.assertIsNotNone(row.finished_at)
        self.assertIsNotNone(row.started_at)
        self.assertEqual(row.attempts, 1)
        self.assertIsNone(row.error)
        (result,) = self.results()
        self.assertEqual(result.job_task_id, self.ml_id)
        self.assertEqual(result.model_version, self.model.model_version)
        self.assertEqual(result.params_hash, self.model.params_hash)
        self.assertEqual(len(result.params_hash), 64)
        self.assertEqual(result.payload["tier"], "ml")
        self.assertEqual(len(result.payload["tox21"]["signals"]), 12)
        self.assertIn("logs", result.payload["solubility"])
        json.dumps(result.payload)

    def test_the_stored_payload_is_what_the_predictor_returns_for_the_stored_molecule(self):
        self.run_task()
        (result,) = self.results()
        self.assertEqual(result.payload, self.model.predict_smiles(ASPIRIN).payload)

    def test_other_tiers_are_left_alone(self):
        self.run_task()
        self.assertEqual(self.row(self.dl_id).status, "queued")

    def test_nfr006_running_the_same_message_twice_stores_one_result(self):
        self.assertEqual(self.run_task(), tasks.OUTCOME_DONE)
        self.assertEqual(self.run_task(), tasks.OUTCOME_IGNORED)
        self.assertEqual(len(self.results()), 1)
        self.assertEqual(self.row().attempts, 1)

    def test_nfr006_a_redelivered_running_task_is_run_again_to_one_result(self):
        self.set_status("running")  # a worker died after claiming the task
        self.assertEqual(self.run_task(), tasks.OUTCOME_DONE)
        self.assertEqual(self.row().status, "done")
        self.assertEqual(len(self.results()), 1)

    def test_an_existing_result_row_is_replaced_not_duplicated(self):
        with Session(self.engine) as session:
            session.add(Result(job_task_id=self.ml_id, model_version="old", params_hash="0" * 64, payload={}))
            session.commit()
        self.set_status("running")
        self.run_task()
        (result,) = self.results()
        self.assertEqual(result.model_version, self.model.model_version)
        self.assertEqual(result.payload["tier"], "ml")


class NotOursTests(TaskCase):
    def test_a_cancelled_task_is_not_started(self):
        self.set_status("cancelled")
        self.assertEqual(self.run_task(), tasks.OUTCOME_IGNORED)
        self.assertEqual(self.row().status, "cancelled")
        self.assertEqual(self.results(), [])

    def test_final_states_are_never_reopened(self):
        for status in ("done", "failed", "skipped"):
            with self.subTest(status=status):
                self.set_status(status)
                self.assertEqual(self.run_task(), tasks.OUTCOME_IGNORED)
                self.assertEqual(self.row().status, status)
        self.assertEqual(self.results(), [])

    def test_a_task_of_another_tier_sent_to_this_queue_is_ignored(self):
        self.assertEqual(self.run_task(self.dl_id), tasks.OUTCOME_IGNORED)
        self.assertEqual(self.row(self.dl_id).status, "queued")

    def test_an_unknown_or_malformed_task_id_is_ignored(self):
        self.assertEqual(self.run_task(uuid.uuid4()), tasks.OUTCOME_IGNORED)
        self.assertEqual(tasks.run_ml_task("not-a-uuid", lambda: self.model), tasks.OUTCOME_IGNORED)
        self.assertEqual(tasks.run_ml_task(None, lambda: self.model), tasks.OUTCOME_IGNORED)

    def test_a_task_cancelled_while_computing_drops_its_result(self):
        def cancel_then_give_model():
            self.set_status("cancelled")
            return self.model

        self.assertEqual(self.run_task(model=cancel_then_give_model), tasks.OUTCOME_DROPPED)
        self.assertEqual(self.row().status, "cancelled")
        self.assertEqual(self.results(), [])

    def test_a_late_failure_does_not_overwrite_a_cancelled_task(self):
        def cancel_then_fail():
            self.set_status("cancelled")
            raise ModelLoadError("gone")

        self.assertEqual(self.run_task(model=cancel_then_fail), tasks.OUTCOME_FAILED)
        self.assertEqual(self.row().status, "cancelled")


class FailureTests(TaskCase):  # task failures are stored, never HTTP errors
    def assert_failed(self, code):
        row = self.row()
        self.assertEqual((row.status, row.error), ("failed", code))
        self.assertIsNotNone(row.finished_at)
        self.assertEqual(self.results(), [])

    def test_missing_or_changed_model_files_give_model_unavailable(self):
        def broken():
            raise ModelLoadError("SHA-256 mismatch: model file was refused.")

        self.assertEqual(self.run_task(model=broken), tasks.OUTCOME_FAILED)
        self.assert_failed(tasks.ERROR_MODEL_UNAVAILABLE)

    def test_an_unreadable_model_file_gives_model_unavailable(self):
        def unreadable():
            raise PermissionError("denied")

        self.run_task(model=unreadable)
        self.assert_failed(tasks.ERROR_MODEL_UNAVAILABLE)

    def test_the_soft_time_limit_gives_timeout(self):
        class Slow:
            def predict_smiles(self, _smiles):
                raise SoftTimeLimitExceeded()

        self.run_task(model=Slow())
        self.assert_failed(tasks.ERROR_TIMEOUT)

    def test_the_soft_time_limit_while_loading_gives_timeout(self):
        def slow_load():
            raise SoftTimeLimitExceeded()

        self.run_task(model=slow_load)
        self.assert_failed(tasks.ERROR_TIMEOUT)

    def test_any_other_error_gives_compute_error(self):
        class Broken:
            def predict_smiles(self, _smiles):
                raise ZeroDivisionError("boom")

        self.run_task(model=Broken())
        self.assert_failed(tasks.ERROR_COMPUTE)

    def test_a_stored_structure_that_cannot_be_parsed_gives_compute_error(self):
        with Session(self.engine) as session:
            session.get(Molecule, self.molecule_id).canonical_smiles = "C(C)(C)(C)(C)C"
            session.commit()
        self.run_task()
        self.assert_failed(tasks.ERROR_COMPUTE)

    def test_a_task_whose_molecule_row_is_missing_gives_compute_error(self):
        with Session(self.engine) as session:
            session.get(Molecule, self.molecule_id).canonical_smiles = ""
            session.commit()
        self.run_task()
        self.assert_failed(tasks.ERROR_COMPUTE)

    def test_sec007_logs_hold_the_task_id_and_exception_class_but_not_the_structure(self):
        class Leaky:
            def predict_smiles(self, smiles):
                raise ValueError(f"cannot handle {smiles}")

        with self.assertLogs("msp.worker.ml", level="INFO") as logs:
            self.run_task(model=Leaky())
        text = "\n".join(logs.output)
        self.assertIn(str(self.ml_id), text)
        self.assertIn("ValueError", text)
        self.assertNotIn(ASPIRIN, text)
        self.assertNotIn("cannot handle", text)

    def test_a_failed_task_can_be_redelivered_only_if_it_is_queued_again(self):
        self.run_task(model=lambda: (_ for _ in ()).throw(ModelLoadError("x")))
        self.assertEqual(self.run_task(), tasks.OUTCOME_IGNORED)  # failed is final here
        self.set_status("queued")                                  # a retry re-queues it
        self.assertEqual(self.run_task(), tasks.OUTCOME_DONE)


class CeleryWiringTests(unittest.TestCase):
    def test_the_task_is_registered_under_the_agreed_name(self):
        self.assertEqual(TASK_NAMES["ml"], "msp.ml.run")
        self.assertIn("msp.ml.run", celery_app.tasks)
        self.assertEqual(tasks.ml_run.name, "msp.ml.run")

    def test_the_entry_point_takes_one_id_and_returns_the_outcome(self):
        with patch.object(tasks, "run_ml_task", return_value="done") as run:
            self.assertEqual(tasks.ml_run.run("abc"), "done")
        run.assert_called_once_with("abc")

    def test_start_up_load_failure_is_logged_not_raised(self):
        with patch.object(tasks, "get_predictor", side_effect=ModelLoadError("no files")):
            with self.assertLogs("msp.worker.ml", level="ERROR") as logs:
                tasks._load_models_at_start()
        self.assertIn("ModelLoadError", "\n".join(logs.output))

    def test_start_up_loads_the_models_when_they_are_fine(self):
        with patch.object(tasks, "get_predictor") as get:
            tasks._load_models_at_start()
        get.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
