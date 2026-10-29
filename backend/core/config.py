"""settings from environment (all open TBD-C12 values).

Owner: Ayushi. Requirements: SEC-004, SEC-003.
"""
"""Settings from environment (all open TBD-C12 values).

Owner: Ayushi. Requirements: SEC-004, SEC-003.
Defaults below are PROVISIONAL until TBD-C8/C9/C10/C12 are decided.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _s(name, default=""):
    return os.environ.get(name) or default


def _i(name, default):
    return int(os.environ.get(name) or default)


def _f(name, default):
    return float(os.environ.get(name) or default)


@dataclass(frozen=True)
class Settings:
    database_url: str
    redis_url: str
    jwt_secret: str
    frontend_origin: str
    max_smiles_len: int
    parse_timeout_s: int
    task_max_retries: int
    ml_soft_limit_s: int
    ml_hard_limit_s: int
    dl_soft_limit_s: int
    dl_hard_limit_s: int
    qml_soft_limit_s: int
    qml_hard_limit_s: int
    session_idle_min: int
    max_failed_logins: int
    lockout_min: int
    rate_limit: str
    pubchem_rps: int
    cache_ttl_hours: int
    qml_concurrency: int
    ad_tanimoto_min: float
    qml_conv_threshold_mha: float
    triage_rules_path: str
    targets_path: str
    models_dir: str

    @classmethod
    def from_env(cls):
        return cls(
            database_url=_s("DATABASE_URL"),
            redis_url=_s("REDIS_URL", "redis://localhost:6379/0"),
            jwt_secret=_s("JWT_SECRET"),
            frontend_origin=_s("FRONTEND_ORIGIN", "http://localhost:3000"),
            max_smiles_len=_i("MAX_SMILES_LEN", 500),
            parse_timeout_s=_i("PARSE_TIMEOUT_S", 5),
            task_max_retries=_i("TASK_MAX_RETRIES", 3),
            ml_soft_limit_s=_i("ML_SOFT_LIMIT_S", 30),
            ml_hard_limit_s=_i("ML_HARD_LIMIT_S", 60),
            dl_soft_limit_s=_i("DL_SOFT_LIMIT_S", 60),
            dl_hard_limit_s=_i("DL_HARD_LIMIT_S", 120),
            qml_soft_limit_s=_i("QML_SOFT_LIMIT_S", 300),
            qml_hard_limit_s=_i("QML_HARD_LIMIT_S", 600),
            session_idle_min=_i("SESSION_IDLE_MIN", 30),
            max_failed_logins=_i("MAX_FAILED_LOGINS", 5),
            lockout_min=_i("LOCKOUT_MIN", 15),
            rate_limit=_s("RATE_LIMIT", "60/minute"),
            pubchem_rps=_i("PUBCHEM_RPS", 5),
            cache_ttl_hours=_i("CACHE_TTL_HOURS", 24),
            qml_concurrency=_i("QML_CONCURRENCY", 1),
            ad_tanimoto_min=_f("AD_TANIMOTO_MIN", 0.3),
            qml_conv_threshold_mha=_f("QML_CONV_THRESHOLD_MHA", 1.6),
            triage_rules_path=_s("TRIAGE_RULES_PATH", str(ROOT / "backend/config/triage_rules.yaml")),
            targets_path=_s("TARGETS_PATH", str(ROOT / "backend/config/targets.yaml")),
            models_dir=_s("MODELS_DIR", str(ROOT / "models")),
        )


def load_yaml_safe(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_triage_rules(settings=None):
    settings = settings or Settings.from_env()
    return load_yaml_safe(settings.triage_rules_path)


def load_targets(settings=None):
    settings = settings or Settings.from_env()
    return load_yaml_safe(settings.targets_path)