"""User queries.

Owner: Ayushi. Requirements: FR-041.
"""
from sqlalchemy import select

from backend.db.models import User


def get_by_email(session, email):
    """Return the user with this email, or None."""
    return session.execute(select(User).where(User.email == email)).scalar_one_or_none()


def create(session, email, password_hash, role="researcher"):
    """Insert a user (new accounts are Researchers) and flush so the id is set."""
    user = User(email=email, password_hash=password_hash, role=role)
    session.add(user)
    session.flush()
    return user
