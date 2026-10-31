# Security validation report (MSP-39)

Owner: Ayushi (test manager and security lead). Scope: SEC-001 to SEC-008 and the OWASP Top 10 checks
from the Master Test Plan 5.1. Pass condition for the freeze: no open High or Critical finding, or an
agreed exception written in `docs/deviation-log.md`.

Run on 2026-10-31, Windows laptop, Python 3.10.11, branch `dev-ayushi`. Deployed environments (Docker
image and CI) use Python 3.11.

## 1. How to reproduce

```
python -m unittest tests.unit.test_msp15 tests.unit.test_msp16 tests.unit.test_msp17 tests.unit.test_msp18 tests.unit.test_msp33 tests.unit.test_api_skeleton tests.unit.molecules.test_validate_route tests.security.test_security
bandit -r backend -ll
pip-audit -r requirements.txt
```
Secret scan and `npm audit` are CI jobs on every pull request (jobs: Secret scan, Frontend). pip-audit also
runs in the CI Backend job on Python 3.11.

## 2. Results of the automated runs

| Run | Result |
|---|---|
| Unit and security tests listed above | 107 tests, all passed |
| Bandit (`-ll`, Medium and above), 3091 lines of backend code | No issues identified. 0 High, 0 Medium. 1 Low issue exists below the reporting level (list it with `bandit -r backend`) |
| pip-audit on the Python 3.10 laptop | 39 advisories, all in Pillow 12.3.0 (fixed in 13.0.0, which needs Python 3.11). See F-01 |
| pip-audit in CI (Python 3.11) | Required check on every pull request; the Backend job passed on the merged pull requests |
| CI job Secret scan | Passed on the merged pull requests |

## 3. SEC requirements

| Req | Statement | Evidence | Result |
|---|---|---|---|
| SEC-001 | Authenticate before protected operations | `test_msp18` test_every_protected_route_rejects_anonymous_requests: every non-public route returns 401 | Pass |
| SEC-002 | Restrict records by permission | `test_security` test_sec002_cross_user_requests_get_403_and_change_nothing; `test_msp18`; `test_msp33`. Another user's or an administrator's request gets 403 and nothing changes | Pass |
| SEC-003 | Secrets out of source control | `test_security` StaticChecks: .env not tracked, .env.example holds no secret values, no secret patterns in source; CI Secret scan | Pass |
| SEC-004 | Input, size and time limits | `test_security`: injection strings rejected with 422, oversized body gives 413, wrong types rejected; `test_validate_route` (length, parse timeout) | Pass |
| SEC-005 | API rate limit, HTTP 429 | `test_msp18` test_rate_limit_gives_429_with_retry_after (with Retry-After header) | Pass |
| SEC-006 | HTTPS with TLS in deployed environments | Config review of `deploy/nginx/nginx.conf`: HTTP redirects to HTTPS (301), port 443 listens with ssl, certificate and key paths set. OWASP ZAP baseline and TLS handshake check not run (need a deployed stack) | Partly verified |
| SEC-007 | No molecular structures in logs | `test_security` test_sec007_no_structures_in_logs_or_audit_log: valid and invalid SMILES sent to submit and validate, none appear in log output or audit rows | Pass |
| SEC-008 | Trusted model artifacts, pinned dependencies | `test_msp15` (SHA-256 mismatch and pickle refused); StaticChecks (all requirements pinned, no unsafe calls in backend); Bandit clean at Medium and above; pip-audit clean in CI (F-01 applies to Python 3.10 only) | Pass |

## 4. OWASP Top 10 (2021)

| Item | Check | Evidence | Result |
|---|---|---|---|
| A01 Broken access control | Other users' jobs, admin routes, cancel | SEC-001 and SEC-002 tests | Pass |
| A02 Cryptographic failures | Argon2id hashes, JWT secret from environment, TLS | `test_msp16` (argon2id), StaticChecks, nginx config review. TLS handshake not run | Pass (handshake not run) |
| A03 Injection | SQL, command and script strings in the SMILES field and target id; SQLAlchemy statements only | test_sec004_injection_strings_are_rejected_safely, tables and rows unchanged | Pass |
| A04 Insecure design | Lockout, rate limit, idle expiry | `test_msp16`, `test_msp17`, `test_msp18` | Pass |
| A05 Security misconfiguration | Debug off, strict CORS, no stack traces | test_owasp_a05_debug_off_and_cors_is_strict; `test_api_skeleton` (500 leaks nothing) | Pass |
| A06 Vulnerable components | pip-audit, npm audit | pip-audit clean in CI on Python 3.11; npm audit runs in CI job Frontend; local advisories F-01 | Pass (CI) |
| A07 Identification and authentication failures | Generic login error, lockout, session expiry, logout | `test_msp16`, `test_msp17` | Pass |
| A08 Software and data integrity failures | Model SHA-256, no pickle, safe YAML | `test_msp15`, StaticChecks, Bandit | Pass |
| A09 Logging and monitoring failures | Audit log for logins, lockouts, submissions, cancellations; no structures in logs | `test_msp16`, `test_msp33`, test_sec007 | Pass |
| A10 Server-side request forgery | Backend calls only fixed hosts | Search client: hosts are constants (`BASE` in pubchem.py and chembl.py). The PubChem query is URL-quoted with `safe=''`, the ChEMBL query is passed as a params dict, and image URLs built from upstream ids are returned to the client, not fetched | Pass |

## 5. Findings log

| ID | Finding | Severity | Owner of the code | Status | Re-test |
|---|---|---|---|---|---|
| F-01 | Pillow 12.3.0 has 39 advisories, fixed in 13.0.0 (needs Python 3.11+). Pillow is a transitive dependency of rdkit. The Docker image and CI use Python 3.11, where the fixed version is installed; developer laptops on Python 3.10 still get 12.3.0 | Low for deployed environments | Shared (requirements.txt) | Mitigated: deployed environments use Python 3.11. Recommend that developers use Python 3.11 | pip-audit in the CI Backend job |
| F-02 | OWASP ZAP baseline and TLS handshake check not run; the nginx config is correct but the live endpoint has not been probed | Low | Abheetha (deploy/) | Open, recorded as a deviation | Run against the Docker stack if time allows before the freeze |

## 6. Conclusion

Seven of eight SEC requirements pass; SEC-006 is partly verified (configuration reviewed, live TLS and ZAP
checks not run). All ten OWASP items pass on the checks that were run. No High or Critical finding is open.
F-02 is recorded as a deviation. Re-run this report after any defect fix before the freeze.
