"""Worker killed, QML timeout, model failures: the other tiers' results survive.

Written for MSP-40 (Abheetha); the folder owner, Ayushi, reviews.
Requirements: BR-004, QA-REL, NFR-005, FR-058..FR-060.
Each test breaks one tier and checks that the other tier's stored result is still served and
the job still ends in a state a user can read.
"""
# pylint: disable=missing-function-docstring,missing-class-docstring,too-few-public-methods
import unittest
import uuid

from celery.exceptions import SoftTimeLimitExceeded

from backend.api.molecules.search_routes import SERVICE_CONFIG_KEY
from backend.registry.loader import ModelLoadError
from backend.services.search_client.errors import UpstreamHTTPError, UpstreamTimeout
from backend.services.search_client.service import SearchService
from backend.workers.agg.tasks import finalise_in_session
from tests.mocks.flow_harness import (FakeMlModel, FlowTestCase, RaisingModel, StubQmlModel,
                                      serving)


class QmlTimeoutTests(FlowTestCase):
    def test_qml_timeout_fails_only_the_qml_task(self):  # FR-059, BR-004
        self.submit()
        self.run_worker("ml", serving(FakeMlModel()))
        outcome = self.run_worker("qml", serving(RaisingModel(SoftTimeLimitExceeded())))
        self.assertEqual(outcome, ("failed", 0))
        job = self.status()
        self.assertEqual((job["by_tier"]["qml"]["status"], job["by_tier"]["qml"]["error"]),
                         ("failed", "TIMEOUT"))
        self.assertIsNone(job["by_tier"]["qml"]["result"])
        self.assertEqual(job["by_tier"]["ml"]["status"], "done")
        self.assertEqual(job["by_tier"]["ml"]["result"]["payload"]["logS"], -2.5)
        self.assertEqual(job["status"], "partial")  # the job ends, with what it has

    def test_qml_timeout_before_ml_finishes_still_keeps_the_ml_result(self):  # NFR-005
        self.submit()
        self.run_worker("qml", serving(RaisingModel(SoftTimeLimitExceeded())))
        self.assertEqual(self.status()["status"], "running")  # ML is still pending
        self.run_worker("ml", serving(FakeMlModel()))
        job = self.status()
        self.assertEqual(job["status"], "partial")
        self.assertEqual(job["by_tier"]["ml"]["result"]["payload"]["logS"], -2.5)


class WorkerKilledTests(FlowTestCase):
    def test_killed_worker_leaves_the_task_running_and_other_results_available(self):  # QA-REL
        self.submit()
        # A worker killed after it claimed the task leaves it "running" and writes nothing.
        self.assertEqual(self.claim_and_die("qml"), 1)
        self.run_worker("ml", serving(FakeMlModel()))
        job = self.status()
        self.assertEqual(job["by_tier"]["qml"]["status"], "running")
        self.assertEqual(job["by_tier"]["ml"]["result"]["payload"]["logS"], -2.5)
        self.assertEqual(job["status"], "running")

    def test_redelivered_message_finishes_the_job_after_a_worker_died(self):  # FR-060
        self.submit()
        self.claim_and_die("qml")  # the first worker dies here
        self.run_worker("ml", serving(FakeMlModel()))
        self.assertEqual(self.run_worker("qml", serving(StubQmlModel())), ("done", 0))  # redelivery
        job = self.status()
        self.assertEqual(job["status"], "complete")
        self.assertEqual(self.task_row("qml").attempts, 2)  # the second attempt is on record
        self.assertEqual(self.result_count("qml"), 1)

    def test_worker_that_never_comes_back_is_failed_by_the_hard_time_limit(self):  # FR-059
        self.submit()
        self.claim_and_die("qml")
        self.run_worker("ml", serving(FakeMlModel()))
        self.make_overdue("qml")
        job = self.status()  # reading the status applies the hard limit
        self.assertEqual((job["by_tier"]["qml"]["status"], job["by_tier"]["qml"]["error"]),
                         ("failed", "TIMEOUT"))
        self.assertEqual(job["status"], "partial")
        self.assertEqual(job["by_tier"]["ml"]["result"]["payload"]["logS"], -2.5)


class OtherTierFailureTests(FlowTestCase):
    def test_ml_failure_does_not_invalidate_the_qml_result(self):  # BR-004
        self.submit()
        outcome = self.run_worker("ml", serving(RaisingModel(ValueError("bad descriptor"))))
        self.assertEqual(outcome, ("failed", 0))
        self.run_worker("qml", serving(StubQmlModel()))
        job = self.status()
        self.assertEqual((job["by_tier"]["ml"]["status"], job["by_tier"]["ml"]["error"]),
                         ("failed", "COMPUTE_ERROR"))
        self.assertEqual(job["by_tier"]["qml"]["status"], "done")
        self.assertIn("stub", job["by_tier"]["qml"]["result"]["payload"]["label"])
        self.assertEqual(job["status"], "partial")

    def test_missing_model_files_fail_the_tier_with_model_unavailable(self):  # empty models/
        self.submit()

        def no_models(_target):
            raise ModelLoadError("model files are missing")

        self.assertEqual(self.run_worker("ml", no_models), ("failed", 0))
        self.run_worker("qml", serving(StubQmlModel()))
        job = self.status()
        self.assertEqual((job["by_tier"]["ml"]["status"], job["by_tier"]["ml"]["error"]),
                         ("failed", "MODEL_UNAVAILABLE"))
        self.assertEqual(job["by_tier"]["qml"]["status"], "done")
        self.assertNotIn("model files are missing", str(job))  # no internal detail reaches the user

    def test_every_tier_failing_ends_the_job_as_failed_with_no_results(self):
        self.submit()
        self.run_worker("ml", serving(RaisingModel(ValueError("x"))))
        self.run_worker("qml", serving(RaisingModel(SoftTimeLimitExceeded())))
        job = self.status()
        self.assertEqual(job["status"], "failed")
        self.assertTrue(all(task["result"] is None for task in job["tasks"]))


class JobSummaryTests(FlowTestCase):
    """The aggregator (MSP-31) builds a summary from whatever the tiers left behind."""

    def finalise(self):
        with self.scope() as session:
            return finalise_in_session(session, uuid.UUID(self.job_id))

    def test_job_still_gets_a_summary_when_a_tier_failed(self):  # BR-004, FR-033
        self.submit()
        self.run_worker("ml", serving(FakeMlModel()))
        self.run_worker("qml", serving(RaisingModel(SoftTimeLimitExceeded())))
        # "exists" is also fine: the worker lifecycle may already have queued the summary
        self.assertIn(self.finalise(), ("created", "exists"))
        job = self.status()
        self.assertIsNotNone(job["summary"])
        self.assertEqual(job["status"], "partial")
        self.assertEqual(job["by_tier"]["ml"]["result"]["payload"]["logS"], -2.5)  # ML survives

    def test_summary_is_built_once_when_finalise_runs_twice(self):  # BR-005 (idempotent)
        self.submit()
        self.run_worker("ml", serving(FakeMlModel()))
        self.run_worker("qml", serving(RaisingModel(SoftTimeLimitExceeded())))
        self.finalise()
        self.assertEqual(self.finalise(), "exists")

    def test_no_summary_while_a_killed_worker_has_not_finished(self):  # QA-REL
        self.submit()
        self.claim_and_die("qml")  # QML stays "running"
        self.run_worker("ml", serving(FakeMlModel()))
        self.assertEqual(self.finalise(), "waiting")  # not every task is final yet
        self.assertEqual(self.status()["status"], "running")


class _MemoryCache:
    def get(self, source, query):  # pylint: disable=unused-argument
        return None

    def set(self, source, query, payload):
        pass


class _NoLimiter:
    def acquire(self, service="PubChem", max_wait=5.0):
        pass


def _failing_pubchem(mode):
    """A PubChem client that fails the way search_mocks describes: timeout, http_500, empty."""
    def client(_query):
        if mode == "timeout":
            raise UpstreamTimeout("PubChem", "PubChem timed out.")
        if mode == "http_500":
            raise UpstreamHTTPError("PubChem", "PubChem returned HTTP 500.")
        return []  # "empty": the service turns this into an UpstreamEmpty error
    return client


class PubchemOutageTests(FlowTestCase):
    """PubChem down: search says so by name, but screening by direct SMILES still works."""

    def use_pubchem(self, mode):
        service = SearchService(_MemoryCache(), _NoLimiter(),
                                {"pubchem": _failing_pubchem(mode), "chembl": lambda q: []})
        self.client.application.config[SERVICE_CONFIG_KEY] = service

    def search(self):
        return self.client.get("/api/molecules/search?query=aspirin&source=pubchem",
                               headers=self.headers)

    def test_outage_names_pubchem_in_the_error(self):  # FR-008, R02
        for mode, status in (("timeout", 504), ("http_500", 502), ("empty", 502)):
            with self.subTest(mode=mode):
                self.use_pubchem(mode)
                resp = self.search()
                self.assertEqual(resp.status_code, status)
                error = resp.get_json()["error"]
                self.assertIn("PubChem", error["message"])
                self.assertEqual(error["details"][0]["service"], "PubChem")

    def test_direct_smiles_job_still_completes_with_a_summary(self):  # FR-061, BR-004
        self.use_pubchem("timeout")
        self.assertEqual(self.search().status_code, 504)  # the outage is real
        self.submit()  # entering SMILES directly calls no outside service
        self.run_worker("ml", serving(FakeMlModel()))
        self.run_worker("qml", serving(StubQmlModel()))
        with self.scope() as session:
            finalise_in_session(session, uuid.UUID(self.job_id))
        job = self.status()
        self.assertEqual(job["status"], "complete")
        self.assertIsNotNone(job["summary"])
        self.assertEqual(job["by_tier"]["ml"]["result"]["payload"]["logS"], -2.5)


if __name__ == "__main__":
    unittest.main()
