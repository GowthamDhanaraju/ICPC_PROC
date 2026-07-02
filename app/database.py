import datetime
import uuid
from typing import Generator
from sqlalchemy import create_engine, Column, String, Float, DateTime, ForeignKey, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from app.config import settings

Base = declarative_base()

class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    candidate_id = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False, default="QUEUED")  # QUEUED, PROCESSING, COMPLETED, FAILED
    source_video_s3_uri = Column(String(1024), nullable=False)
    enrollment_photo_s3_uri = Column(String(1024), nullable=True)
    overall_score = Column(Float, nullable=True)
    webhook_url = Column(String(1024), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Unique constraint for S3 idempotency: combination of video source to prevent double processing
    # Note: For S3 trigger, we can map this via source_video_s3_uri
    
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
            "violations": [v.to_dict() for v in self.violations]
        }

class Violation(Base):
    __tablename__ = "violations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    type = Column(String(100), nullable=False)  # MULTIPLE_FACES, NO_FACE_DETECTED, etc.
    start_ts = Column(Float, nullable=False)  # seconds
    end_ts = Column(Float, nullable=False)    # seconds
    duration = Column(Float, nullable=False)  # seconds
    confidence = Column(Float, nullable=False)
    evidence_frame_s3_uri = Column(String(1024), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    job = relationship("Job", back_populates="violations")

    def to_dict(self):
        # Format timestamps as HH:MM:SS for the webhook payload
        def format_time(seconds: float) -> str:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int((seconds - int(seconds)) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d}"

        return {
            "type": self.type,
            "start_ts": format_time(self.start_ts),
            "end_ts": format_time(self.end_ts),
            "start_seconds": self.start_ts,
            "end_seconds": self.end_ts,
            "duration": self.duration,
            "confidence": round(self.confidence, 4),
            "evidence_frame_s3_uri": self.evidence_frame_s3_uri
        }

# Engine and Session Setup
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
