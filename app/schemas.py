"""
Pydantic request/response schemas for the Batch Video Proctoring Pipeline API.
Separated from main.py to keep route handlers clean and allow reuse by SDK.
"""
from pydantic import BaseModel, ConfigDict, field_validator
from typing import Optional, List


class JobCreate(BaseModel):
    """Payload for submitting a new proctoring session."""
    candidate_id: str
    video_s3_uri: str
    enrollment_photo_s3_uri: Optional[str] = None
    webhook_url: Optional[str] = None

    @field_validator("video_s3_uri")
    @classmethod
    def must_be_s3_or_local(cls, v: str) -> str:
        if not (v.startswith("s3://") or v.startswith("/")):
            raise ValueError("video_s3_uri must be an S3 URI (s3://...) or absolute local path")
        return v


class ViolationResponse(BaseModel):
    """Single detected violation event in the proctoring timeline."""
    model_config = ConfigDict(from_attributes=True)

    type: str
    start_ts: str
    end_ts: str
    start_seconds: float
    end_seconds: float
    duration: float
    confidence: float
    evidence_frame_s3_uri: Optional[str] = None


class JobSubmitResponse(BaseModel):
    """Response returned immediately after a session is submitted."""
    job_id: str
    status: str
    message: Optional[str] = None


class JobResponse(BaseModel):
    """Full status and results for a proctoring job."""
    model_config = ConfigDict(from_attributes=True)

    job_id: str
    candidate_id: str
    status: str
    source_video_s3_uri: str
    enrollment_photo_s3_uri: Optional[str] = None
    overall_score: Optional[float] = None
    webhook_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    violations: List[ViolationResponse] = []


class HealthResponse(BaseModel):
    """Health / liveness check response."""
    status: str
    version: str


class ReadyResponse(BaseModel):
    """Readiness check response — verifies all dependencies are reachable."""
    status: str
    database: str
    queue: str
