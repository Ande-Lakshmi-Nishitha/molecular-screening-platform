"""Unit tests for the QML worker task: lifecycle, failures, retry, wiring (MSP-27).

Covers FR-021, FR-029, FR-058..060, SEC-002, SEC-007, NFR-006. The database is an in-memory SQLite
one; the stub backend is the real one (it needs no models).

Run from the repo root with:  python -m unittest discover -s tests -t .
"""
# pylint: disable=missing-function-docstring,missing-class-docstring
import logging
import unittest
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.celery_app import TASK_NAMES, celery_app
from backend.db.models import Base, Job, JobTask, Molecule, Result, User
from backend.registry.loader import ModelLoadError
from backend.workers.qml import predictor, tasks

SETTINGS = SimpleNamespace(qml_conv_threshold_mha=1.6)


class TaskCase(unittest.TestCase):
    quiet = logging.NullHandler()

    def setUp(self):
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
        self.model = predictor.get_predictor(settings=SETTINGS)
        self.molecule_id = None
        self.qml_id, self.ml_id = self.new_job("CCO")

    def new_job(self, smiles):
        with Session(self.engine) as session:
            user = session.execute(select(User)).scalars().first() or User(email="q@example.com", password_hash="x")
            session.add(user)
            session.flush()
            molecule = Molecule(inchikey=uuid.uuid4().hex[:27].upper(), canonical_smiles=smiles,
                                formula="X", heavy_atoms=3)
            session.add(molecule)
            session.flush()
            job = Job(molecule_id=molecule.id, user_id=user.id)
            session.add(job)
            session.flush()
            qml = JobTask(job_id=job.id, tier="qml", status="queued")
            other = JobTask(job_id=job.id, tier="ml", status="queued")
            session.add_all([qml, other])
            session.commit()
            return qml.id, other.id

    def run_task(self, task_id=None, model=None):
        get_model = model if callable(model) else (lambda _t: model or self.model)
        return tasks.run_qml_task(str(task_id or self.qml_id), get_model)

    def row(self, task_id=None):
        with Session(self.engine) as session:
            task = session.get(JobTask, task_id or self.qml_id)
            return task.status, task.error, task.attempts

    def results(self):
        with Session(self.engine) as session:
            return session.execute(select(Result)).scalars().all()


class SuccessTests(TaskCase):
    def test_a_queued_task_finishes_with_one_stored_labelled_stub_result(self):
        self.assertEqual(self.run_task(), tasks.OUTCOME_DONE)
        self.assertEqual(self.row()[0], "done")
        (result,) = self.results()
        self.assertEqual(result.model_version, "qml_stub@1.0.0")
        self.assertEqual(len(result.params_hash), 64)
        self.assertTrue(result.payload["is_stub"])
        self.assertEqual(result.payload["electronic_stability"]["unit"], "eV")

    def test_nfr006_running_the_same_message_twice_stores_one_result(self):
        self.run_task()
        self.assertEqual(self.run_task(), tasks.OUTCOME_IGNORED)
        self.assertEqual(len(self.results()), 1)

    def test_other_tiers_and_bad_ids_are_left_alone(self):
        self.assertEqual(self.run_task(self.ml_id), tasks.OUTCOME_IGNORED)
        self.assertEqual(self.row(self.ml_id)[0], "queued")
        self.assertEqual(tasks.run_qml_task("nope"), tasks.OUTCOME_IGNORED)

    def test_a_cancelled_task_is_not_started_and_a_late_result_is_dropped(self):
        def cancel_then_give_model(_target):
            with Session(self.engine) as session:
                session.get(JobTask, self.qml_id).status = "cancelled"
                session.commit()
            return self.model

        self.assertEqual(self.run_task(model=cancel_then_give_model), tasks.OUTCOME_DROPPED)
        self.assertEqual((self.row()[0], self.results()), ("cancelled", []))
        self.assertEqual(self.run_task(), tasks.OUTCOME_IGNORED)


class FailureTests(TaskCase):
    def assert_failed(self, code):
        status, error, _ = self.row()
        self.assertEqual((status, error), ("failed", code))
        self.assertEqual(self.results(), [])

    def test_an_unknown_backend_fails_as_model_unavailable(self):
        def no_backend(_target):
            raise ModelLoadError("Unknown quantum backend.")
        self.assertEqual(self.run_task(model=no_backend), tasks.OUTCOME_FAILED)
        self.assert_failed(tasks.ERROR_MODEL_UNAVAILABLE)

    def test_fr021_a_molecule_that_does_not_qualify_fails_as_compute_error(self):
        task_id, _ = self.new_job("CCCCCCCCCC")   # 10 heavy atoms
        self.assertEqual(self.run_task(task_id), tasks.OUTCOME_FAILED)
        self.assertEqual(self.row(task_id)[:2], ("failed", tasks.ERROR_COMPUTE))

    def test_the_soft_time_limit_is_timeout(self):
        class Slow:
            def predict_smiles(self, _smiles):
                raise SoftTimeLimitExceeded()
        self.assertEqual(self.run_task(model=Slow()), tasks.OUTCOME_FAILED)
        self.assert_failed(tasks.ERROR_TIMEOUT)

    def test_sec007_log_lines_never_hold_the_structure(self):
        with self.assertLogs("msp.worker.qml", level="ERROR") as logs:
            task_id, _ = self.new_job("CCCCCCCCCC")
            self.run_task(task_id)
        text = "\n".join(logs.output)
        self.assertIn("ValueError", text)
        self.assertNotIn("CCCCCCCCCC", text)

    def test_a_short_fault_is_retried_then_succeeds(self):
        class Flaky:
            calls = 0

            def predict_smiles(self, smiles):
                Flaky.calls += 1
                if Flaky.calls == 1:
                    raise ConnectionError("lost")
                return predictor.get_predictor(settings=SETTINGS).predict_smiles(smiles)
        self.assertEqual(tasks.run_qml_task_with_delay(str(self.qml_id), lambda _t: Flaky()), (tasks.OUTCOME_RETRY, 5))
        self.assertEqual(self.row()[0], "queued")
        self.assertEqual(tasks.run_qml_task(str(self.qml_id), lambda _t: Flaky()), tasks.OUTCOME_DONE)
        self.assertEqual(len(self.results()), 1)


class CeleryWiringTests(unittest.TestCase):
    def test_the_task_is_registered_under_the_agreed_name(self):
        self.assertEqual(TASK_NAMES["qml"], "msp.qml.run")
        self.assertIn("msp.qml.run", celery_app.tasks)
        self.assertEqual(tasks.qml_run.name, "msp.qml.run")

    def test_the_entry_point_takes_one_id_and_returns_the_outcome(self):
        with patch.object(tasks, "run_qml_task_with_delay", return_value=("done", 0)) as run:
            self.assertEqual(tasks.qml_run.run("abc"), "done")
        run.assert_called_once_with("abc")

    def test_the_entry_point_retries_then_returns_the_final_outcome(self):
        with patch.object(tasks, "run_qml_task_with_delay", side_effect=[("retry", 5), ("done", 0)]) as run:
            result = tasks.qml_run.apply(args=["abc"])
        self.assertEqual((result.get(), run.call_count), ("done", 2))


if __name__ == "__main__":
    unittest.main()
