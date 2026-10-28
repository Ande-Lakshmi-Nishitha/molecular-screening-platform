# Repository structure

```
molecular-screening-platform/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── nightly.yml
│   │   └── pr-checks.yml
│   ├── CODEOWNERS
│   ├── pull_request_template.md
│   └── team.json
├── backend/
│   ├── api/
│   │   ├── admin/
│   │   │   └── routes.py
│   │   ├── auth/
│   │   │   ├── lockout.py
│   │   │   ├── passwords.py
│   │   │   ├── rbac.py
│   │   │   ├── routes.py
│   │   │   ├── service.py
│   │   │   └── sessions.py
│   │   ├── jobs/
│   │   │   ├── cancel.py
│   │   │   ├── report.py
│   │   │   ├── reuse.py
│   │   │   ├── routes.py
│   │   │   └── status.py
│   │   ├── models_info/
│   │   │   └── routes.py
│   │   ├── molecules/
│   │   │   ├── routes.py
│   │   │   └── search_routes.py
│   │   ├── app.py
│   │   ├── errors.py
│   │   ├── health.py
│   │   ├── openapi.py
│   │   └── schemas.py
│   ├── config/
│   │   ├── targets.yaml
│   │   └── triage_rules.yaml
│   ├── core/
│   │   ├── audit.py
│   │   ├── config.py
│   │   ├── labels.py
│   │   └── logging_filter.py
│   ├── db/
│   │   ├── migrations/
│   │   │   └── README.md
│   │   ├── repositories/
│   │   │   ├── jobs.py
│   │   │   ├── models_repo.py
│   │   │   └── users.py
│   │   ├── models.py
│   │   └── session.py
│   ├── registry/
│   │   ├── cards/
│   │   │   └── README.md
│   │   └── loader.py
│   ├── services/
│   │   ├── molecule/
│   │   │   ├── eligibility.py
│   │   │   ├── parse_process.py
│   │   │   └── standardise.py
│   │   └── search_client/
│   │       ├── cache.py
│   │       ├── chembl.py
│   │       └── pubchem.py
│   ├── workers/
│   │   ├── agg/
│   │   │   ├── flags.py
│   │   │   ├── tasks.py
│   │   │   └── triage.py
│   │   ├── common/
│   │   │   └── base_task.py
│   │   ├── dl/
│   │   │   ├── baseline.py
│   │   │   ├── ensemble.py
│   │   │   ├── graph.py
│   │   │   ├── neighbours.py
│   │   │   └── tasks.py
│   │   ├── ml/
│   │   │   ├── alerts.py
│   │   │   ├── domain.py
│   │   │   ├── featuriser.py
│   │   │   ├── solubility.py
│   │   │   ├── tasks.py
│   │   │   └── tox21.py
│   │   └── qml/
│   │       ├── backend.py
│   │       ├── convergence.py
│   │       ├── fragments.py
│   │       ├── references.py
│   │       ├── stub_backend.py
│   │       └── tasks.py
│   ├── Dockerfile
│   ├── celery_app.py
│   └── wsgi.py
├── deploy/
│   ├── backup/
│   │   ├── backup.sh
│   │   └── restore.sh
│   ├── certs/
│   │   └── README.md
│   └── nginx/
│       └── nginx.conf
├── docs/
│   ├── api/
│   ├── decisions/
│   │   └── decision-log.md
│   ├── demo/
│   ├── sad/
│   ├── sprint-reports/
│   ├── srs/
│   ├── test-cases/
│   ├── test-plan/
│   ├── traceability/
│   ├── GIT_WORKFLOW.md
│   ├── README.md
│   └── deviation-log.md
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── api/
│   │   │   ├── client.js
│   │   │   └── errors.js
│   │   ├── components/
│   │   │   ├── JobStatus.jsx
│   │   │   ├── ResultBlock.jsx
│   │   │   └── Warnings.jsx
│   │   ├── pages/
│   │   │   ├── Admin.jsx
│   │   │   ├── Compare.jsx
│   │   │   ├── History.jsx
│   │   │   ├── JobTracker.jsx
│   │   │   ├── Login.jsx
│   │   │   ├── ModelInfo.jsx
│   │   │   ├── Register.jsx
│   │   │   ├── Results.jsx
│   │   │   ├── Search.jsx
│   │   │   └── Submit.jsx
│   │   ├── App.jsx
│   │   └── index.jsx
│   ├── README.md
│   └── package.json
├── models/
│   └── README.md
├── tests/
│   ├── failure_injection/
│   │   └── test_failures.py
│   ├── fixtures/
│   │   ├── chembl_responses/
│   │   ├── pubchem_responses/
│   │   ├── README.md
│   │   └── golden_molecules.py
│   ├── integration/
│   │   └── test_submit_flow.py
│   ├── mocks/
│   │   └── pubchem_chembl.py
│   ├── model_validation/
│   │   └── test_qa_acc.py
│   ├── performance/
│   │   └── locustfile.py
│   ├── security/
│   │   └── test_security.py
│   ├── system/
│   │   ├── api/
│   │   │   └── test_api_contract.py
│   │   └── selenium/
│   │       └── test_ui_flows.py
│   └── unit/
│       ├── agg/
│       │   └── test_triage.py
│       ├── auth/
│       │   └── test_auth.py
│       ├── jobs/
│       │   └── test_job_status.py
│       ├── molecules/
│       │   └── test_standardise.py
│       ├── registry/
│       │   └── test_loader.py
│       ├── workers_dl/
│       │   └── test_dl.py
│       ├── workers_ml/
│       │   └── test_ml.py
│       └── workers_qml/
│           └── test_qml.py
├── training/
│   ├── data/
│   │   └── README.md
│   ├── evaluate.py
│   ├── export_models.py
│   ├── train_gnn.py
│   ├── train_solubility.py
│   └── train_tox21.py
├── .dockerignore
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── README.md
├── docker-compose.yml
├── requirements-dev.txt
└── requirements.txt
```

Owners: backend/api/auth, jobs, workers/agg, db, core/config, core/audit, tests/security, tests/failure_injection, docs = Ayushi. backend/workers/ml|dl|qml, services/molecule, registry, core/labels, training = Nishitha. frontend, deploy, backend/api/app|errors|schemas|health|admin, search client, tests/system|integration|performance = Abheetha.
