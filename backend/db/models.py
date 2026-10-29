import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (JSON, Boolean, DateTime, ForeignKey, Integer, String,
                        Text, UniqueConstraint, Uuid)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

JSONType = JSON().with_variant(JSONB(), "postgresql")

TIERS = ("ml", "dl", "qml")
TASK_STATUSES = ("queued", "running", "done", "failed", "skipped", "cancelled")
FINAL_STATUSES = ("done", "failed", "skipped", "cancelled")
ROLES = ("researcher", "administrator")


def _now():
    return datetime.now(timezone.utc)


def _uuid():
    return uuid.uuid4()


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="researcher")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("users.id"), nullable=True)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    detail: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)  # never structures (SEC-007)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Molecule(Base):
    __tablename__ = "molecules"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    inchikey: Mapped[str] = mapped_column(String(27), unique=True, nullable=False)
    canonical_smiles: Mapped[str] = mapped_column(Text, nullable=False)
    formula: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    heavy_atoms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    source_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    molecule_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("molecules.id"), nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)  # ownership is on the job
    target_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    options: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
    rules_version: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    summary: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    cancelled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class JobTask(Base):
    __tablename__ = "job_tasks"
    __table_args__ = (UniqueConstraint("job_id", "tier", name="uq_job_tier"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id"), nullable=False)
    tier: Mapped[str] = mapped_column(String(10), nullable=False)
    status: Mapped[str] = mapped_column(String(15), nullable=False, default="queued")
    skip_reason: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    error: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class Result(Base):
    __tablename__ = "results"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    job_task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("job_tasks.id"), unique=True, nullable=False)  # upsert key
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    params_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ModelRecord(Base):
    __tablename__ = "models"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_uuid)
    tier: Mapped[str] = mapped_column(String(10), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    metrics: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
    artifact_location: Mapped[str] = mapped_column(String(255), nullable=False)
    data_provenance: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    