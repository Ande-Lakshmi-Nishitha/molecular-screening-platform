"""Tests for the model-tier task life on top of Ayushi's base_task (MSP-26).

Covers FR-058..060 (retry with a growing delay, stored failure reasons), FR-029, SEC-007. The
database is an in-memory SQLite one and the model is a small fake, so no torch is needed.

Run from the repo root with:  python -m unittest discover -s tests -t .
"""
# pylint: disable=missing-function-docstring,missing-class-docstring
import logging
import unittest
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from celery.exceptions import Retry, SoftTimeLimitExceeded
from sqlalchemy import create_engine, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.db.models import Base, Job, JobTask, Molecule, Result, User
from backend.registry.loader import ModelLoadError
from backend.workers.common import base_task, task_lifecycle as tl
from backend.workers.ml import tasks as ml_tasks

try:
    import torch  # noqa: F401  # pylint: disable=unused-import
    from backend.workers.dl import tasks as dl_tasks
except ImportError:  # CI installs requirements.txt only: the DL entry point needs torch
    dl_tasks = None

log = logging.getLogger("msp.test.lifecycle")


class FakeModel:
    def __init__(self, error=None):
        self.error, self.seen = error, []

    def predict_smiles(self, smiles):
        self.seen.append(smiles)
        if self.error:
            raise self.error
        return SimpleNamespace(model_version="1.0.0", params_hash="h", payload={"ok": True})


class LifecycleCase(unittest.TestCase):
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

        self.scope = scope
        quiet = logging.NullHandler()
        log.addHandler(quiet)
        self.addCleanup(log.removeHandler, quiet)
        with Session(engine) as session:
            user = User(email="l@example.com", password_hash="x")
            session.add(user)
            session.flush()
            molecule = Molecule(inchikey="BSYNRYMUTXBXSQ-UHFFFAOYSA-N", canonical_smiles="CC(=O)O",
                                formula="C2H4O2", heavy_atoms=4)
            session.add(molecule)
            session.flush()
            job = Job(molecule_id=molecule.id, user_id=user.id, target_id="EGFR")
            session.add(job)
            session.flush()
            task = JobTask(job_id=job.id, tier="dl", status="queued")
            other = JobTask(job_id=job.id, tier="ml", status="queued")
            session.add_all([task, other])
            session.commit()
            self.task_id, self.other_id = task.id, other.id

    def run_it(self, model, task_id=None, max_retries=3, seen_targets=None):
        def get_model(target):
            if seen_targets is not None:
                seen_targets.append(target)
            if isinstance(model, Exception):
                raise model
            return model
        return tl.run_task(str(task_id or self.task_id), "dl", get_model, log, self.scope, max_retries)

    def row(self, task_id=None):
        with Session(self.engine) as session:
            task = session.get(JobTask, task_id or self.task_id)
            return task.status, task.error, task.attempts

    def results(self):
        with Session(self.engine) as session:
            return session.execute(select(Result)).scalars().all()


class OutcomeTests(LifecycleCase):
    def test_success_stores_one_result_and_reads_the_job_target(self):
        targets, model = [], FakeModel()
        self.assertEqual(self.run_it(model, seen_targets=targets), (tl.OUTCOME_DONE, 0))
        self.assertEqual(targets, ["EGFR"])
        self.assertEqual(model.seen, ["CC(=O)O"])
        self.assertEqual(self.row()[0], "done")
        self.assertEqual(len(self.results()), 1)

    def test_never_claimed_is_ignored_but_finished_after_cancel_is_dropped(self):
        with Session(self.engine) as session:
            session.get(JobTask, self.task_id).status = "cancelled"
            session.commit()
        model = FakeModel()
        self.assertEqual(self.run_it(model), (tl.OUTCOME_IGNORED, 0))
        self.assertEqual(model.seen, [])

        with Session(self.engine) as session:
            session.get(JobTask, self.task_id).status = "queued"
            session.commit()

        def cancel_then_give_model(_target):
            with Session(self.engine) as session:
                session.get(JobTask, self.task_id).status = "cancelled"
                session.commit()
            return FakeModel()
        outcome = tl.run_task(str(self.task_id), "dl", cancel_then_give_model, log, self.scope, 3)
        self.assertEqual(outcome, (tl.OUTCOME_DROPPED, 0))
        self.assertEqual((self.row()[0], self.results()), ("cancelled", []))

    def test_another_tier_and_bad_ids_are_ignored_without_touching_the_row(self):
        self.assertEqual(self.run_it(FakeModel(), task_id=self.other_id), (tl.OUTCOME_IGNORED, 0))
        self.assertEqual(self.row(self.other_id), ("queued", None, 0))
        self.assertEqual(self.run_it(FakeModel(), task_id=uuid.uuid4()), (tl.OUTCOME_IGNORED, 0))
        self.assertEqual(tl.run_task("nope", "dl", lambda t: None, log, self.scope, 3), (tl.OUTCOME_IGNORED, 0))


class FailureReasonTests(LifecycleCase):
    def test_model_problems_are_stored_as_model_unavailable_and_are_not_retried(self):
        for error in (ModelLoadError("gone"), OSError("disk")):
            with self.subTest(error=type(error).__name__):
                with Session(self.engine) as session:
                    task = session.get(JobTask, self.task_id)
                    task.status, task.error = "queued", None
                    session.commit()
                self.assertEqual(self.run_it(error), (tl.OUTCOME_FAILED, 0))
                self.assertEqual(self.row()[:2], ("failed", "MODEL_UNAVAILABLE"))

    def test_soft_time_limit_is_timeout_and_other_errors_are_compute_error(self):
        self.assertEqual(self.run_it(FakeModel(SoftTimeLimitExceeded())), (tl.OUTCOME_FAILED, 0))
        self.assertEqual(self.row()[:2], ("failed", "TIMEOUT"))
        with Session(self.engine) as session:
            task = session.get(JobTask, self.task_id)
            task.status, task.error = "queued", None
            session.commit()
        self.assertEqual(self.run_it(FakeModel(ValueError("x"))), (tl.OUTCOME_FAILED, 0))
        self.assertEqual(self.row()[:2], ("failed", "COMPUTE_ERROR"))

    def test_a_missing_molecule_is_a_compute_error(self):
        with Session(self.engine) as session:
            job = session.get(Job, session.get(JobTask, self.task_id).job_id)
            session.get(Molecule, job.molecule_id).canonical_smiles = ""
            session.commit()
        self.assertEqual(self.run_it(FakeModel()), (tl.OUTCOME_FAILED, 0))
        self.assertEqual(self.row()[:2], ("failed", "COMPUTE_ERROR"))

    def test_decide_failure_uses_a_reason_carried_by_the_error_but_still_retries_transient_errors(self):
        self.assertEqual(base_task.decide_failure(tl.TaskFailure("MODEL_UNAVAILABLE"), 1, 3),
                         ("fail", "MODEL_UNAVAILABLE", 0))
        self.assertEqual(base_task.decide_failure(SoftTimeLimitExceeded(), 1, 3), ("fail", "TIMEOUT", 0))
        self.assertEqual(base_task.decide_failure(ConnectionError("x"), 1, 3), ("retry", None, 5))
        self.assertEqual(base_task.decide_failure(ValueError("x"), 1, 3), ("fail", "COMPUTE_ERROR", 0))
        odd = ValueError("x")
        odd.reason = 7                      # not a text reason: ignored
        self.assertEqual(base_task.decide_failure(odd, 1, 3), ("fail", "COMPUTE_ERROR", 0))


class RetryTests(LifecycleCase):
    def test_a_short_fault_requeues_with_growing_delays_then_fails_after_the_limit(self):
        error = OperationalError("select", {}, ConnectionError("lost"))
        delays = []
        for attempt in (1, 2, 3):
            outcome, delay = self.run_it(FakeModel(error))
            delays.append((outcome, delay))
            self.assertEqual(self.row()[0:3:2], ("queued", attempt))
        self.assertEqual(delays, [("retry", 5), ("retry", 10), ("retry", 20)])
        self.assertEqual(self.run_it(FakeModel(error)), (tl.OUTCOME_FAILED, 0))   # fourth attempt: no more retries
        self.assertEqual(self.row(), ("failed", "COMPUTE_ERROR", 4))

    def test_a_retried_task_can_then_succeed_with_one_result(self):
        self.assertEqual(self.run_it(FakeModel(ConnectionError("x")))[0], tl.OUTCOME_RETRY)
        self.assertEqual(self.run_it(FakeModel()), (tl.OUTCOME_DONE, 0))
        self.assertEqual((self.row()[0], len(self.results())), ("done", 1))

    def test_max_retries_comes_from_the_settings_when_not_given(self):
        with patch.object(tl.Settings, "from_env", return_value=SimpleNamespace(task_max_retries=0)):
            outcome = tl.run_task(str(self.task_id), "dl", lambda t: FakeModel(ConnectionError("x")), log, self.scope)
        self.assertEqual(outcome, (tl.OUTCOME_FAILED, 0))

    def test_sec007_log_lines_hold_the_task_id_and_exception_class_but_not_the_structure(self):
        with self.assertLogs(log, level="DEBUG") as logs:
            self.run_it(FakeModel(ValueError("CC(=O)O is secret")))
        text = "\n".join(logs.output)
        self.assertIn("ValueError", text)
        self.assertIn(str(self.task_id), text)
        self.assertNotIn("CC(=O)O", text)


class RetryOrReturnTests(unittest.TestCase):
    def test_a_retry_outcome_asks_celery_to_retry_after_the_delay(self):
        task = MagicMock()
        task.retry.return_value = Retry()
        with patch.object(tl.Settings, "from_env", return_value=SimpleNamespace(task_max_retries=3)):
            with self.assertRaises(Retry):
                tl.retry_or_return(task, tl.OUTCOME_RETRY, 10)
        task.retry.assert_called_once_with(countdown=10, max_retries=4)

    def test_other_outcomes_are_returned_unchanged(self):
        task = MagicMock()
        for outcome in (tl.OUTCOME_DONE, tl.OUTCOME_FAILED, tl.OUTCOME_DROPPED, tl.OUTCOME_IGNORED):
            self.assertEqual(tl.retry_or_return(task, outcome, 0), outcome)
        task.retry.assert_not_called()


class EntryPointTests(unittest.TestCase):
    """The real Celery tasks, run eagerly: a retry outcome makes Celery run the task again."""

    def test_both_entry_points_retry_then_return_the_final_outcome(self):
        cases = [(ml_tasks, "ml_run", "run_ml_task_with_delay")]
        if dl_tasks is not None:
            cases.append((dl_tasks, "dl_run", "run_dl_task_with_delay"))
        for module, entry, runner in cases:
            with self.subTest(entry=entry):
                with patch.object(module, runner, side_effect=[("retry", 5), ("done", 0)]) as run:
                    result = getattr(module, entry).apply(args=["abc"])
                self.assertEqual((result.get(), run.call_count), ("done", 2))

    @unittest.skipIf(dl_tasks is None, "torch is not installed")
    def test_the_dl_entry_point_has_the_agreed_name(self):
        self.assertEqual((ml_tasks.ml_run.name, dl_tasks.dl_run.name), ("msp.ml.run", "msp.dl.run"))


if __name__ == "__main__":
    unittest.main()
