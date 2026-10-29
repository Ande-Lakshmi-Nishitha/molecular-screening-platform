from alembic import context
from sqlalchemy import create_engine

from backend.core.config import Settings
from backend.db.models import Base

target_metadata = Base.metadata


def run_migrations_online():
    engine = create_engine(Settings.from_env().database_url)
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run_migrations_online()