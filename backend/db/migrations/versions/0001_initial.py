"""initial schema (SRS B.5 + models.sha256, jobs.cancelled_at, results.params_hash)"""
from alembic import op

from backend.db.models import Base

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    Base.metadata.create_all(bind=op.get_bind())


def downgrade():
    Base.metadata.drop_all(bind=op.get_bind())