from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.core.config import Settings

_engine = None
_factory = None


def get_engine():
    global _engine, _factory
    if _engine is None:
        url = Settings.from_env().database_url
        if not url:
            raise RuntimeError("DATABASE_URL is not set")
        _engine = create_engine(url, pool_pre_ping=True)
        _factory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


@contextmanager
def session_scope():
    get_engine()
    session = _factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    