"""Audit log helper (FR-065).

Owner: Ayushi. Rows are added to the caller's open session, so they commit in the
same transaction as the action. Never put molecular structures in `detail` (SEC-007).
"""
from backend.db.models import AuditLog


def record(session, event_type, user_id=None, detail=None):
    """Add an audit row to the open session and return it."""
    row = AuditLog(event_type=event_type, user_id=user_id, detail=detail)
    session.add(row)
    return row
