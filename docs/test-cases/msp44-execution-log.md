# MSP-44 execution log (tester: Ayushi)

Scope: F-01, F-02, F-07, F-08, F-14 (226 cases, 105 High). Status on 2026-10-31: in progress.

## Automated evidence collected

| Date | Run | Result |
|---|---|---|
| 2026-10-31 | `python -m unittest discover -s tests/unit/molecules -t .` (validate, validate route, standardise, eligibility, molfile, search) | 136 tests, all passed, 1 skipped |

One earlier run the same day reported 1 error that did not reproduce on re-run. It is recorded here as a possible intermittent failure, not as a defect.

## Not done yet

- Case-by-case results in the workbook `MSP_Test_Cases_1.xlsx`: the workbook is not in the repository and has not been recorded yet.
- System-level cases (S): need the Docker stack.
- Cases that need the results screen (MSP-37 is not merged): blocked until it is.
- Cases marked `Pending Cn`: stay Pending until the TBD-C value is decided. They are never counted as passed.
