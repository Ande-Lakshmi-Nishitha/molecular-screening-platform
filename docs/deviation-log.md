# Deviation log
Every difference between what was built and the SRS, SAD or test plan, with the reason.

| Date | Document | Deviation | Reason |
| --- | --- | --- | --- |

- MSP-39 / SEC-006: OWASP ZAP baseline and TLS handshake check were not run, because they need a deployed stack. The nginx configuration was reviewed instead (HTTP to HTTPS redirect, TLS on port 443). Recorded as finding F-02 in docs/test-plan/security-validation-report.md.

- 2026-11-02 | SRS 3.4 / FR-031, FR-032 (MSP-30): job status and partial results are one route, `GET /api/jobs/{job_id}`. The SRS endpoint table lists `/api/jobs/{id}/status` and `/api/jobs/{id}/results`. Reason: one response carries every tier's state and any finished result, so the tracker needs a single poll to show partial results.
- 2026-11-02 | SRS 3.4 / FR-036 (MSP-37): job history is served by `GET /api/jobs` (paged), which the SRS endpoint table does not list. Reason: FR-036 needs a list of the caller's own jobs and the table has no endpoint for it.
- 2026-11-02 | SRS 3.4 / FR-039 (MSP-38): report export is `GET /api/jobs/{job_id}/report` (JSON and PDF), not in the SRS endpoint table. Reason: the table omits the export endpoint that FR-039 requires.
- 2026-11-02 | SRS 3.4 / FR-047 (MSP-17): logout is `POST /api/auth/logout`, not in the SRS endpoint table. Reason: the table omits the endpoint that FR-047 requires.
- 2026-11-02 | SRS 4.10 / FR-050, FR-062: no administration routes are registered. The Admin page shows service health from `/healthz` and `/readyz` only. Per-tier queue length and worker health (FR-062) and user management (FR-050) are not implemented. Reason: feature 4.10 is Medium priority in the SRS and was descoped to protect the High-priority features.
- 2026-11-02 | Test plan / MSP-41, QA-CMP: the Selenium suite covers the login flow only (login form, register link, redirect when not logged in, valid login, wrong password). Submit, track and cancel are not automated. Reason: MSP-41 is a Should story, second in the cut order. The submit, track and cancel flows are covered by the manual UI cases in the test runs (MSP-42 to MSP-44).
