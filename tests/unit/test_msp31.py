import unittest
import uuid
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.api.jobs.status import build_status
from backend.celery_app import celery_app
from backend.core.config import load_triage_rules
from backend.db.models import Base, Job, JobTask, Molecule, Result, User
from backend.workers.agg import flags
from backend.workers.agg import tasks as agg_tasks
from backend.workers.agg import triage

RULES = load_triage_rules()


def ml_payload(logs=-3.0, width=1.0, probs=None, domain=None):
    payload = {"tier": "ml",
               "solubility": {"logs": logs, "interval": {"low": logs - width / 2, "high": logs + width / 2, "level": 0.9},
                              "unit": "logS", "limitation": "x", "model_version": "1"},
               "tox21": {"signals": {name: {"probability": p, "label": "x"}
                                     for name, p in (probs or {"NR-AR": 0.1}).items()},
                         "unit": "probability", "limitation": "x", "model_version": "1", "label": "x"}}
    if domain is not None:
        payload["domain"] = {"in_domain": domain}
    return payload


def dl_payload(mean=7.5, sd=0.3, base=7.4):
    return {"tier": "dl", "target": {"id": "EGFR"}, "activity": {"mean": mean, "sd": sd, "n_models": 5},
            "baseline": {"value": base, "algorithm": "x"}}


def qml_payload(gap=5.0, converged=True, stub=True):
    return {"tier": "qml", "is_stub": stub, "electronic_stability": {"metric": "homo_lumo_gap", "value": gap, "unit": "eV"},
            "convergence": {"converged": converged}}


def task(tier, status="done", payload=None, skip=None, error=None):
    return {"tier": tier, "status": status, "skip_reason": skip, "error": error, "payload": payload}


def ml_task(**kw):
    return task("ml", payload=ml_payload(**kw))


def dl_task(**kw):
    return task("dl", payload=dl_payload(**kw))


def qml_task(**kw):
    return task("qml", payload=qml_payload(**kw))


class RuleTests(unittest.TestCase):  # FR-033, BR-005
    def dim(self, tasks, name, rules=RULES):
        return triage.build_triage(tasks, rules)["dimensions"][name]

    def test_rules_file_is_complete_and_versioned(self):
        triage.validate_rules(RULES)
        self.assertTrue(RULES["version"])
        with self.assertRaises(ValueError):
            triage.validate_rules({"version": "x"})

    def test_developability_from_logs(self):
        for logs, expected in ((-3.0, "green"), (-4.0, "green"), (-5.0, "amber"), (-6.0, "amber"), (-7.0, "red")):
            self.assertEqual(self.dim([ml_task(logs=logs)], "developability")["status"], expected, logs)

    def test_safety_from_highest_tox21_probability(self):
        for probs, expected in (({"A": 0.1, "B": 0.2}, "green"), ({"A": 0.5}, "amber"),
                                ({"A": 0.7}, "red"), ({"A": 0.9, "B": 0.1}, "red")):
            self.assertEqual(self.dim([ml_task(probs=probs)], "safety_signals")["status"], expected, probs)
        raised = self.dim([ml_task(probs={"A": 0.9, "B": 0.1})], "safety_signals")["inputs"]["assays_raised"]
        self.assertEqual(raised, ["A"])

    def test_target_activity_from_potency(self):
        for mean, expected in ((8.0, "green"), (7.0, "green"), (6.5, "amber"), (5.0, "red")):
            self.assertEqual(self.dim([dl_task(mean=mean)], "target_activity")["status"], expected, mean)

    def test_electronic_stability_from_gap(self):
        for gap, expected in ((5.0, "green"), (3.0, "amber"), (1.0, "red")):
            self.assertEqual(self.dim([qml_task(gap=gap)], "electronic_stability")["status"], expected, gap)

    def test_unconverged_qml_is_excluded(self):
        result = self.dim([qml_task(converged=False)], "electronic_stability")
        self.assertEqual((result["status"], result["reason"]), ("unavailable", "NOT_CONVERGED"))

    def test_stub_can_be_excluded_by_rule(self):
        rules = {**RULES, "electronic_stability": {**RULES["electronic_stability"], "exclude_stub_backend": True}}
        result = self.dim([qml_task()], "electronic_stability", rules)
        self.assertEqual((result["status"], result["reason"]), ("unavailable", "STUB_BACKEND"))
        self.assertTrue(self.dim([qml_task()], "electronic_stability")["inputs"]["is_stub"])

    def test_confidence_checks_and_worst_colour_wins(self):
        self.assertEqual(self.dim([ml_task(), dl_task(), qml_task()], "confidence")["status"], "green")
        self.assertEqual(self.dim([ml_task(width=3.0)], "confidence")["status"], "amber")
        self.assertEqual(self.dim([dl_task(sd=1.5)], "confidence")["status"], "red")
        self.assertEqual(self.dim([dl_task(mean=7.0, base=5.0)], "confidence")["status"], "amber")
        self.assertEqual(self.dim([ml_task(domain=False)], "confidence")["status"], "amber")
        self.assertEqual(self.dim([ml_task(width=3.0), dl_task(sd=1.5)], "confidence")["status"], "red")
        one = [ml_task(), task("dl", "failed", error="COMPUTE_ERROR")]
        two = [task("ml", "failed", error="COMPUTE_ERROR"), task("dl", "failed", error="TIMEOUT")]
        self.assertEqual(self.dim(one, "confidence")["status"], "amber")
        self.assertEqual(self.dim(two, "confidence")["status"], "red")

    def test_skipped_tiers_do_not_lower_confidence(self):
        tasks = [ml_task(), task("dl", "skipped", skip="NO_TARGET_SELECTED"), task("qml", "skipped", skip="NO_FRAGMENT")]
        self.assertEqual(self.dim(tasks, "confidence")["status"], "green")

    def test_tiers_that_did_not_finish_show_unavailable_with_a_reason(self):
        tasks = [ml_task(), task("dl", "skipped", skip="NO_TARGET_SELECTED")]
        result = triage.build_triage(tasks, RULES)
        self.assertEqual(result["dimensions"]["target_activity"]["reason"], "NO_TARGET_SELECTED")
        self.assertEqual(result["tiers"]["dl"], {"status": "skipped", "reason": "NO_TARGET_SELECTED"})
        for state, reason in (("failed", "COMPUTE_ERROR"), ("cancelled", "CANCELLED"), ("running", "NOT_FINISHED")):
            got = self.dim([task("ml", state, error="COMPUTE_ERROR" if state == "failed" else None)], "developability")
            self.assertEqual((got["status"], got["reason"]), ("unavailable", reason))

    def test_unknown_payload_shape_does_not_crash(self):
        result = triage.build_triage([task("ml", payload={"x": 1})], RULES)
        self.assertEqual(result["dimensions"]["developability"]["reason"], "NO_DATA")
        self.assertEqual(result["dimensions"]["safety_signals"]["reason"], "NO_DATA")

    def test_summary_carries_versions_and_limitation(self):
        summary = triage.build_triage([ml_task()], RULES)
        self.assertEqual(summary["rules_version"], RULES["version"])
        self.assertEqual(set(summary["dimensions"]), {"developability", "safety_signals", "target_activity",
                                                       "electronic_stability", "confidence"})
        self.assertIn("generated_at", summary)
        self.assertIn("limitation", summary)
        if triage.LIMITATION is None:
            self.skipTest("TRIAGE_LIMITATION is not defined in core/labels.py yet")
        self.assertEqual(summary["limitation"], triage.LIMITATION)


class FlagTests(unittest.TestCase):  # FR-034
    def codes(self, tasks):
        return [item["code"] for item in triage.build_triage(tasks, RULES)["flags"]]

    def test_potent_but_poorly_soluble(self):
        self.assertIn("POTENT_POORLY_SOLUBLE", self.codes([dl_task(mean=8.0), ml_task(logs=-7.0)]))

    def test_potent_but_tox21_signal(self):
        self.assertIn("POTENT_TOX21_SIGNAL", self.codes([dl_task(mean=8.0), ml_task(probs={"SR-ARE": 0.6})]))

    def test_good_profile_but_out_of_domain(self):
        self.assertEqual(self.codes([dl_task(mean=8.0), ml_task(domain=False)]), ["GOOD_PROFILE_OUT_OF_DOMAIN"])

    def test_no_flag_when_no_condition_holds(self):
        self.assertEqual(self.codes([dl_task(mean=5.0), ml_task()]), [])
        self.assertEqual(flags.cross_signal_flags({}, {}), [])


class FinaliseTests(unittest.TestCase):  # FR-033, BR-005
    def setUp(self):
        self.engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        with Session(self.engine) as session:
            user = User(email="a@example.com", password_hash="x")
            molecule = Molecule(inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N", canonical_smiles="CCO")
            session.add_all([user, molecule])
            session.commit()
            self.user_id, self.molecule_id = user.id, molecule.id

    @contextmanager
    def scope(self):
        session = Session(self.engine)
        try:
            yield session
            session.commit()
        finally:
            session.close()

    def make_job(self, spec):
        with Session(self.engine) as session:
            job = Job(molecule_id=self.molecule_id, user_id=self.user_id, options={})
            session.add(job)
            session.flush()
            for tier, (status, payload) in spec.items():
                row = JobTask(job_id=job.id, tier=tier, status=status,
                              skip_reason="NO_TARGET_SELECTED" if status == "skipped" else None)
                session.add(row)
                session.flush()
                if status == "done":
                    session.add(Result(job_task_id=row.id, model_version="1.0", params_hash="h", payload=payload))
            session.commit()
            return job.id

    def finalise(self, job_id):
        with self.scope() as session:
            return agg_tasks.finalise_in_session(session, job_id)

    def summary(self, job_id):
        with Session(self.engine) as session:
            return session.get(Job, job_id).summary

    def test_waits_until_every_task_is_final(self):
        job = self.make_job({"ml": ("done", ml_payload()), "dl": ("running", None)})
        self.assertEqual(self.finalise(job), "waiting")
        self.assertIsNone(self.summary(job))

    def test_no_tasks_and_unknown_job(self):
        self.assertEqual(self.finalise(self.make_job({})), "waiting")
        self.assertEqual(self.finalise(uuid.uuid4()), "missing")

    def test_creates_the_summary_once_and_is_idempotent(self):
        job = self.make_job({"ml": ("done", ml_payload()), "dl": ("skipped", None), "qml": ("skipped", None)})
        self.assertEqual(self.finalise(job), "created")
        first = self.summary(job)
        self.assertEqual(first["dimensions"]["developability"]["status"], "green")
        self.assertEqual(self.finalise(job), "exists")
        self.assertEqual(self.summary(job)["generated_at"], first["generated_at"])
        with Session(self.engine) as session:
            stored = session.get(Job, job)
            self.assertEqual(stored.rules_version, RULES["version"])
            self.assertIsNotNone(stored.finished_at)

    def test_status_read_builds_the_summary_when_all_tasks_are_final(self):
        job = self.make_job({"ml": ("done", ml_payload()), "dl": ("skipped", None), "qml": ("skipped", None)})
        with self.scope() as session:
            body = build_status(session, job, self.user_id)
        self.assertEqual(body["summary"]["dimensions"]["developability"]["status"], "green")
        self.assertEqual(body["rules_version"], RULES["version"])

    def test_the_celery_task_is_registered_on_the_agg_queue(self):
        self.assertIn(agg_tasks.TASK_NAME, celery_app.tasks)


if __name__ == "__main__":
    unittest.main()
    