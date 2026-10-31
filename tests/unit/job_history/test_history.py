"""Unit tests for the paged job history (MSP-37; FR-049, SEC-002). Uses an in-memory SQLite database."""
import unittest
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.api.jobs.history_routes import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, PagingError, history_item, parse_paging
from backend.db.models import Base, Job, JobTask, Molecule, User
from backend.db.repositories.job_history import list_jobs_owned

START = datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc)


class HistoryTests(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        self.session = Session(engine)
        self.addCleanup(self.session.close)
        self.alice, self.bob = self.add_user("alice@example.test"), self.add_user("bob@example.test")
        self.molecule = Molecule(inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N", canonical_smiles="CCO", formula="C2H6O", heavy_atoms=3)
        self.session.add(self.molecule)
        self.session.flush()

    def add_user(self, email):
        user = User(email=email, password_hash="x")
        self.session.add(user)
        self.session.flush()
        return user.id

    def add_job(self, user_id, minutes, statuses=("done", "skipped", "failed"), cancelled=False):
        job = Job(molecule_id=self.molecule.id, user_id=user_id, target_id="EGFR",
                  created_at=START + timedelta(minutes=minutes),
                  cancelled_at=START if cancelled else None)
        self.session.add(job)
        self.session.flush()
        for tier, status in zip(("ml", "dl", "qml"), statuses):
            self.session.add(JobTask(job_id=job.id, tier=tier, status=status))
        self.session.flush()
        return job

    def test_returns_only_the_callers_jobs_and_counts_only_theirs(self):
        mine = [self.add_job(self.alice, m) for m in range(3)]
        self.add_job(self.bob, 10)
        self.add_job(self.bob, 11)
        total, rows = list_jobs_owned(self.session, self.alice, 20, 0)
        self.assertEqual(total, 3)
        self.assertEqual({job.id for job, _, _ in rows}, {job.id for job in mine})

    def test_a_user_with_no_jobs_gets_an_empty_page(self):
        self.add_job(self.bob, 1)
        self.assertEqual(list_jobs_owned(self.session, self.alice, 20, 0), (0, []))

    def test_pages_newest_first_without_overlap(self):
        jobs = [self.add_job(self.alice, m) for m in range(5)]
        newest_first = [job.id for job in reversed(jobs)]
        first = [job.id for job, _, _ in list_jobs_owned(self.session, self.alice, 2, 0)[1]]
        second = [job.id for job, _, _ in list_jobs_owned(self.session, self.alice, 2, 2)[1]]
        third = [job.id for job, _, _ in list_jobs_owned(self.session, self.alice, 2, 4)[1]]
        self.assertEqual(first + second + third, newest_first)

    def test_each_row_carries_its_own_tasks(self):
        a = self.add_job(self.alice, 1, statuses=("done", "done", "done"))
        b = self.add_job(self.alice, 2, statuses=("queued", "skipped", "skipped"))
        rows = {job.id: tasks for job, _, tasks in list_jobs_owned(self.session, self.alice, 20, 0)[1]}
        self.assertEqual({t.status for t in rows[a.id]}, {"done"})
        self.assertEqual(sorted(t.status for t in rows[b.id]), ["queued", "skipped", "skipped"])

    def test_history_item_rolls_up_the_status_and_orders_the_tiers(self):
        job = self.add_job(self.alice, 1, statuses=("done", "skipped", "failed"))
        _, ((row_job, molecule, tasks),) = list_jobs_owned(self.session, self.alice, 20, 0)
        item = history_item(row_job, molecule, tasks)
        self.assertEqual(item["job_id"], str(job.id))
        self.assertEqual(item["status"], "partial")
        self.assertEqual(list(item["tiers"]), ["ml", "dl", "qml"])
        self.assertEqual(item["molecule"]["formula"], "C2H6O")
        self.assertEqual(item["target_id"], "EGFR")

    def test_a_cancelled_job_with_nothing_done_reads_cancelled(self):
        self.add_job(self.alice, 1, statuses=("cancelled", "cancelled", "skipped"), cancelled=True)
        _, ((job, molecule, tasks),) = list_jobs_owned(self.session, self.alice, 20, 0)
        self.assertEqual(history_item(job, molecule, tasks)["status"], "cancelled")


class PagingTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(parse_paging({}), (1, DEFAULT_PAGE_SIZE))
        self.assertEqual(parse_paging({"page": "", "page_size": ""}), (1, DEFAULT_PAGE_SIZE))

    def test_accepts_valid_values(self):
        self.assertEqual(parse_paging({"page": "3", "page_size": "50"}), (3, 50))
        self.assertEqual(parse_paging({"page_size": str(MAX_PAGE_SIZE)}), (1, MAX_PAGE_SIZE))

    def test_rejects_bad_values_and_names_the_field(self):
        for args, field in [({"page": "0"}, "page"), ({"page": "-1"}, "page"), ({"page": "x"}, "page"),
                            ({"page_size": "0"}, "page_size"), ({"page_size": str(MAX_PAGE_SIZE + 1)}, "page_size"),
                            ({"page_size": "1.5"}, "page_size")]:
            with self.subTest(args=args):
                with self.assertRaises(PagingError) as caught:
                    parse_paging(args)
                self.assertEqual(caught.exception.field, field)


if __name__ == "__main__":
    unittest.main()
