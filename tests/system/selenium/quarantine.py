"""Quarantine list for flaky UI tests (MSP-41).

A test named here is skipped with its reason, so one flaky test cannot block the suite.
Add a line only after the test failed twice in a row on a clean stack, and say why.
Every entry must also be logged in docs/deviation-log.md and fixed or removed later.

    "LoginFlowTests.test_something": "fails 1 run in 5, tracked in MSP-123",
"""
QUARANTINE = {}
