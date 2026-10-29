import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.core.config import Settings, load_targets, load_triage_rules
from backend.db.models import Base, Molecule

ROOT = Path(__file__).resolve().parents[2]


class ConfigTests(unittest.TestCase):  # FR-053, SEC-004, FR-052
    def test_default_limit_comes_from_config(self):
        with patch.dict(os.environ, {"MAX_SMILES_LEN": ""}):
            self.assertEqual(Settings.from_env().max_smiles_len, 500)

    def test_env_overrides_default(self):
        with patch.dict(os.environ, {"MAX_SMILES_LEN": "123", "LOCKOUT_MIN": "7"}):
            s = Settings.from_env()
            self.assertEqual((s.max_smiles_len, s.lockout_min), (123, 7))

    def test_yaml_files_load_with_safe_loader(self):
        self.assertIn("version", load_triage_rules())
        self.assertIn("targets", load_targets())


class SchemaTests(unittest.TestCase):  # FR-035, BR-008
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)

    def test_tables_and_added_columns(self):
        insp = inspect(self.engine)
        self.assertTrue({"users", "audit_log", "molecules", "jobs", "job_tasks",
                         "results", "models"} <= set(insp.get_table_names()))
        cols = lambda t: {c["name"] for c in insp.get_columns(t)}
        self.assertIn("sha256", cols("models"))
        self.assertIn("cancelled_at", cols("jobs"))
        self.assertIn("params_hash", cols("results"))
        self.assertIn("user_id", cols("jobs"))

    def test_inchikey_is_unique(self):
        with Session(self.engine) as s:
            s.add(Molecule(inchikey="BSYNRYMUTXBXSQ-UHFFFAOYSA-N", canonical_smiles="CC(=O)Oc1ccccc1C(=O)O"))
            s.commit()
            s.add(Molecule(inchikey="BSYNRYMUTXBXSQ-UHFFFAOYSA-N", canonical_smiles="CC(=O)Oc1ccccc1C(=O)O"))
            with self.assertRaises(IntegrityError):
                s.commit()


class MigrationTests(unittest.TestCase):
    def test_migration_runs_from_empty_database(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            url = f"sqlite:///{Path(tmp, 'm.db').as_posix()}"
            with patch.dict(os.environ, {"DATABASE_URL": url}):
                command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
            engine = create_engine(url)
            self.assertIn("jobs", inspect(engine).get_table_names())
            engine.dispose()


if __name__ == "__main__":
    unittest.main()