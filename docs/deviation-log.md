# Deviation log
Every difference between what was built and the SRS, SAD or test plan, with the reason.

| Date | Document | Deviation | Reason |
| --- | --- | --- | --- |

- MSP-39 / SEC-006: OWASP ZAP baseline and TLS handshake check were not run, because they need a deployed stack. The nginx configuration was reviewed instead (HTTP to HTTPS redirect, TLS on port 443). Recorded as finding F-02 in docs/test-plan/security-validation-report.md.
