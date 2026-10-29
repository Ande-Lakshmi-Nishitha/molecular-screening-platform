"""Checks docker-compose.yml against the MSP-11 acceptance criteria.

Covers QA-POR, SEC-003, SEC-006, QA-REC.
"""
# pylint: disable=missing-function-docstring
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {
    "proxy", "api", "worker-ml", "worker-dl", "worker-qml", "worker-agg",
    "redis", "postgres", "db-backup",
}


class ComposeConfigTests(unittest.TestCase):
    """Static checks; the live check is `docker compose up`."""

    @classmethod
    def setUpClass(cls):
        text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        cls.compose = yaml.safe_load(text)
        cls.services = cls.compose["services"]

    def test_qa_por_all_required_services_defined(self):
        self.assertTrue(REQUIRED <= set(self.services))

    def test_sec006_only_proxy_publishes_ports(self):
        published = [n for n, s in self.services.items() if "ports" in s]
        self.assertEqual(published, ["proxy"])

    def test_sec006_proxy_publishes_80_and_443(self):
        self.assertEqual(set(self.services["proxy"]["ports"]), {"80:80", "443:443"})

    def test_two_networks_with_internal_backend(self):
        nets = self.compose["networks"]
        self.assertEqual(set(nets), {"frontend", "backend"})
        self.assertTrue(nets["backend"]["internal"])

    def test_redis_append_only(self):
        command = self.services["redis"]["command"]
        self.assertEqual(command[command.index("--appendonly") + 1], "yes")

    def test_postgres_uses_named_volume(self):
        mount = self.services["postgres"]["volumes"][0]
        self.assertIn("pgdata", self.compose["volumes"])
        self.assertTrue(mount.startswith("pgdata:"))

    def test_every_long_running_service_has_healthcheck(self):
        for name in REQUIRED:
            self.assertIn("healthcheck", self.services[name], name)

    def test_models_mounted_read_only_in_ml_and_dl(self):
        for name in ("worker-ml", "worker-dl"):
            self.assertIn("./models:/models:ro", self.services[name]["volumes"])

    def test_sec003_env_example_has_names_only(self):
        lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        secrets = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")
        for line in lines:
            if line.startswith(secrets):
                self.assertTrue(line.endswith("="), line)

    def test_sec003_env_is_gitignored(self):
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn(".env", ignore)


if __name__ == "__main__":
    unittest.main()
