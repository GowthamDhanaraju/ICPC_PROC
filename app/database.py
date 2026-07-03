"""
SQLAlchemy database models and session management for the Batch Video Proctoring Pipeline.

Production notes:
- PostgreSQL is recommended for multi-worker deployments (configure DATABASE_URL accordingly).
- SQLite remains supported for local development and single-process usage.
- Connection pooling is configured for PostgreSQL; SQLite uses StaticPool.
"""
import datetime
import uuid
import logging
from typing import Generator

from sqlalchemy import (
    create_engine, Column, String, Float, DateTime,
    ForeignKey, Integer, Text, Index, event
)
from sqlalchemy.orm import (
    DeclarativeBase, sessionmaker, relationship, Session
)
from sqlalchemy.pool import StaticPool

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Declarative base (modern SQLAlchemy 2.x style)
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    candidate_id = Column(String(100), nullable=False, index=True)
    status = Column(String(50), nullable=False, default="QUEUED", index=True)
    # Statuses: QUEUED | PROCESSING | COMPLETED | FAILED
    source_video_s3_uri = Column(String(1024), nullable=False)
    enrollment_photo_s3_uri = Column(String(1024), nullable=True)
    overall_score = Column(Float, nullable=True)
    webhook_url = Column(String(1024), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        onupdate=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    violations = relationship("Violation", back_populates="job", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "job_id": self.id,
            "candidate_id": self.candidate_id,
            "status": self.status,
            "source_video_s3_uri": self.source_video_s3_uri,
            "enrollment_photo_s3_uri": self.enrollment_photo_s3_uri,
            "overall_score": self.overall_score,
            "webhook_url": self.webhook_url,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "violations": [v.to_dict() for v in self.violations],
        }


# Compound index for fast idempotency lookups in submit_session
Index(
    "ix_job_idempotency",
    Job.source_video_s3_uri,
    Job.candidate_id,
    Job.status,
)


class Violation(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    type = Column(String(100), nullable=False)
    start_ts = Column(Float, nullable=False)   # seconds
    end_ts = Column(Float, nullable=False)     # seconds
    duration = Column(Float, nullable=False)   # seconds
    confidence = Column(Float, nullable=False)
    evidence_frame_s3_uri = Column(String(1024), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    job = relationship("Job", back_populates="violations")

    def to_dict(self):
        def fmt(s: float) -> str:
            h, rem = divmod(int(s), 3600)
            m, sec = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{sec:02d}"

        return {
            "type": self.type,
            "start_ts": fmt(self.start_ts),
            "end_ts": fmt(self.end_ts),
            "start_seconds": self.start_ts,
            "end_seconds": self.end_ts,
            "duration": self.duration,
            "confidence": round(self.confidence, 4),
            "evidence_frame_s3_uri": self.evidence_frame_s3_uri,
        }


# ---------------------------------------------------------------------------
# Engine & Session factory
# ---------------------------------------------------------------------------
def _build_engine():
    url = settings.DATABASE_URL
    is_sqlite = url.startswith("sqlite")

    if is_sqlite:
        # StaticPool ensures the same in-memory connection is reused across
        # threads in tests. For file-based SQLite, check_same_thread is enough.
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool if ":memory:" in url else None,
        )
        # Enable WAL mode for better concurrent read performance on SQLite
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, _):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()
    else:
        # PostgreSQL — connection pool tuned for multi-worker deployments
        engine = create_engine(
            url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,    # Detect stale connections before use
            pool_recycle=1800,     # Recycle connections every 30 minutes
        )

    return engine


engine = _build_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Creates all database tables if they don't exist."""
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session and ensures cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
