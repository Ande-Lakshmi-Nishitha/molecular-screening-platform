# Decision log

Open SRS values (TBD-C1 to TBD-C12). Provisional values live in `backend/core/config.py` and are read
from environment variables, so changing one never needs a code change. Each person adds the row for
the item they decide. Provisional rows stay "Provisional" until something is measured.

| ID | Item | Value | Status | Owner | Needed before |
|---|---|---|---|---|---|
| TBD-C1 | Library versions to pin | Pinned in `requirements.txt`: Flask 3.1.3, SQLAlchemy 2.0.36, alembic 1.13.3, psycopg2-binary 2.9.10, PyYAML 6.0.2, argon2-cffi 23.1.0, redis 5.0.8, PyJWT 2.15.0, celery 5.4.0, gunicorn 22.0.0, rdkit 2026.9.1. Flask and PyJWT were raised after pip-audit findings. Pillow comes in through rdkit and is not pinned here. | Decided | Ayushi | Every story |
| TBD-C2 | Supported targets with ChEMBL IDs | EGFR (CHEMBL203), BACE1 (CHEMBL4822), HIV-1 protease (CHEMBL243). To be confirmed in `backend/config/targets.yaml`. | Provisional | Nishitha | MSP-25 |
| TBD-C3 | Model architectures and hyperparameters | ML tier: classical models for solubility and Tox21. DL tier: 5-model GNN ensemble per target plus a baseline. Hyperparameters are recorded in each model card. | Provisional | Nishitha | MSP-22, MSP-25 |
| TBD-C4 | Acceptance thresholds | Not set. Recorded as Pending until a baseline run has produced metrics. | Provisional | Nishitha | MSP-22, MSP-43 |
| TBD-C5 | Quantum framework | Qiskit Nature or PennyLane, decided by a short feasibility run. Until then the stub backend stays the quantum tier behind the `QuantumBackend` interface. | Open | Nishitha | MSP-28 only |
| TBD-C6 | Convergence threshold and backend | Convergence threshold 1.6 mHa (config `QML_CONV_THRESHOLD_MHA`). Default active space 4 electrons in 4 orbitals. | Provisional | Nishitha | MSP-28 |
| TBD-C7 | Fragment selection rules | Parked. The fragment path is descoped in v1.0. | Parked | Nishitha | Not needed |
| TBD-C8 | Applicability-domain thresholds | Minimum Tanimoto similarity 0.3 (config `AD_TANIMOTO_MIN`). | Provisional | Nishitha | MSP-24 |
| TBD-C9 | Triage thresholds and cross-tier rules | Rules file `backend/config/triage_rules.yaml`, version 0.1, thresholds still empty. Set in MSP-31. | Provisional | Ayushi | MSP-31 |
| TBD-C10 | Reference hardware and latency targets | Not measured yet. Time limits per tier are config values (see C12). Latency claims wait for a measurement. | Provisional | Nishitha | MSP-23, MSP-26, MSP-41 |
| TBD-C11 | Retention and privacy policy | No code in v1.0. One line in the README (MSP-47). | Decided | Ayushi | MSP-47 |
| TBD-C12 | Operational limits | Max SMILES length 500. Parse timeout 5 s. Task retries 3. ML time limit 30 s soft, 60 s hard. DL 60 s soft, 120 s hard. QML 300 s soft, 600 s hard. Session idle timeout 30 min, absolute session limit 8 h. Lockout after 5 failed logins for 15 min. API rate limit 60 per minute. All are environment variables with these defaults. | Provisional | Ayushi | MSP-13, 16, 17, 18, 20, 33 |