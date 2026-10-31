"""Submit, partial result, complete (stub QML), end to end.

Owner: Abheetha. Requirements: QA-TST, NFR-005, FR-029, FR-030, FR-031, FR-032, QA-REL.
API -> queue message -> worker lifecycle -> database -> status endpoint, in one process
(see tests/mocks/flow_harness.py).
"""
# pylint: disable=missing-function-docstring,missing-class-docstring
import unittest

from tests.mocks.flow_harness import FakeMlModel, FlowTestCase, StubQmlModel, serving


class SubmitFlowTests(FlowTestCase):
    def test_submit_queues_one_message_per_runnable_tier(self):  # FR-029
        body = self.submit()
        tasks = {task["tier"]: task for task in body["tasks"]}
        self.assertEqual(tasks["ml"]["status"], "queued")
        self.assertEqual(tasks["qml"]["status"], "queued")
        self.assertEqual((tasks["dl"]["status"], tasks["dl"]["skip_reason"]),
                         ("skipped", "NO_TARGET_SELECTED"))
        sent = {call.args[1]: str(call.args[0]) for call in self.enqueue.call_args_list}
        self.assertEqual(sent, {"ml": tasks["ml"]["id"], "qml": tasks["qml"]["id"]})

    def test_job_is_queued_until_a_worker_picks_it_up(self):  # FR-030
        self.submit()
        job = self.status()
        self.assertEqual(job["status"], "queued")
        self.assertIsNone(job["by_tier"]["ml"]["result"])

    def test_partial_result_is_visible_while_the_other_tier_is_unfinished(self):  # FR-031, NFR-005
        self.submit()
        self.assertEqual(self.run_worker("ml", serving(FakeMlModel())), ("done", 0))
        job = self.status()
        self.assertEqual(job["status"], "running")
        self.assertEqual(job["by_tier"]["ml"]["status"], "done")
        self.assertEqual(job["by_tier"]["ml"]["result"]["payload"]["logS"], -2.5)
        self.assertEqual(job["by_tier"]["qml"]["status"], "queued")
        self.assertIsNone(job["by_tier"]["qml"]["result"])

    def test_job_is_complete_when_every_runnable_tier_is_done(self):  # FR-030, FR-032
        self.submit()
        self.run_worker("ml", serving(FakeMlModel()))
        self.assertEqual(self.run_worker("qml", serving(StubQmlModel())), ("done", 0))
        job = self.status()
        self.assertEqual(job["status"], "complete")
        ml_payload = job["by_tier"]["ml"]["result"]["payload"]
        self.assertEqual(ml_payload["logS_interval_90"], [-3.1, -1.9])
        self.assertIn("stub", job["by_tier"]["qml"]["result"]["payload"]["label"])
        self.assertEqual(job["by_tier"]["dl"]["status"], "skipped")  # skipped is not a failure
        self.assertEqual(job["by_tier"]["dl"]["skip_reason"], "NO_TARGET_SELECTED")

    def test_result_records_the_model_version_that_produced_it(self):  # QA-REP, FR-035
        self.submit()
        self.run_worker("ml", serving(FakeMlModel()))
        result = self.status()["by_tier"]["ml"]["result"]
        self.assertEqual((result["model_version"], result["params_hash"]),
                         ("ml-test-1.0", "ml-hash"))

    def test_a_message_delivered_twice_stores_one_result(self):  # QA-REL: queue redelivery
        self.submit()
        self.run_worker("ml", serving(FakeMlModel()))
        self.run_worker("ml", serving(FakeMlModel()))
        self.assertEqual(self.result_count("ml"), 1)
        self.assertEqual(self.status()["by_tier"]["ml"]["status"], "done")


if __name__ == "__main__":
    unittest.main()
