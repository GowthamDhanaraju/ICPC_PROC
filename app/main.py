from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel, HttpUrl
from typing import Optional, List
from sqlalchemy.orm import Session
from app.database import init_db, get_db, Job
from app.config import settings
from app.orchestration.queue import global_queue

# Pydantic schemas for request validation
class JobCreate(BaseModel):
    candidate_id: str
    video_s3_uri: str
    enrollment_photo_s3_uri: Optional[str] = None
    webhook_url: Optional[str] = None

class ViolationResponse(BaseModel):
    type: str
    start_ts: str
    end_ts: str
    start_seconds: float
    end_seconds: float
    duration: float
    confidence: float
    evidence_frame_s3_uri: Optional[str] = None

class JobResponse(BaseModel):
    job_id: str
    candidate_id: str
    status: str
    source_video_s3_uri: str
    enrollment_photo_s3_uri: Optional[str] = None
    overall_score: Optional[float] = None
    webhook_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: str
    violations: List[ViolationResponse]

# Initialize FastAPI App
app = FastAPI(
    title="Batch Video Proctoring API",
    description="S3 Event-driven / REST Batch proctoring pipeline analysis engine.",
    version="0.1.0"
)

# Startup & Shutdown Lifecycles
@app.on_event("startup")
def startup_event():
    print("Initializing Database tables...")
    init_db()
    print("Starting background job queue workers...")
    global_queue.start(num_workers=2)

@app.on_event("shutdown")
def shutdown_event():
    print("Stopping background job queue workers...")
    global_queue.stop()

# API Endpoints
@app.post("/v1/sessions", status_code=status.HTTP_201_CREATED)
def submit_session(payload: JobCreate, db: Session = Depends(get_db)):
    """
    Submits a batch video proctoring job.
    Inserts a job entry to database and queues it for background pipeline processing.
    """
    try:
        # Check for idempotency: if active job with same key exists, return it
        existing = db.query(Job).filter(
            Job.source_video_s3_uri == payload.video_s3_uri,
            Job.candidate_id == payload.candidate_id,
            Job.status.in_(["QUEUED", "PROCESSING"])
        ).first()
        
        if existing:
            return {
                "job_id": existing.id,
                "status": existing.status,
                "message": "Job already exists and is queued/processing."
            }

        # Create new Job
        job = Job(
            candidate_id=payload.candidate_id,
            source_video_s3_uri=payload.video_s3_uri,
            enrollment_photo_s3_uri=payload.enrollment_photo_s3_uri,
            webhook_url=payload.webhook_url,
            status="QUEUED"
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        
        # Submit to internal work queue
        global_queue.submit(job.id)
        
        return {
            "job_id": job.id,
            "status": job.status
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit proctoring job: {str(e)}"
        )

@app.get("/v1/sessions/{job_id}", response_model=JobResponse)
def get_session(job_id: str, db: Session = Depends(get_db)):
    """
    Retrieves the status, score, and timeline violations of a proctoring job.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Proctoring job with ID {job_id} not found."
        )
    return job.to_dict()

# Mock target endpoint for local testing webhooks
received_webhooks = []

@app.post("/test/webhook-target", status_code=status.HTTP_200_OK)
def test_webhook_target(payload: dict):
    """
    Local test webhook endpoint to log callback updates.
    """
    print(f"[TEST WEBHOOK RECEIVED] Job {payload.get('job_id')} status: {payload.get('status')} score: {payload.get('overall_score')}")
    received_webhooks.append(payload)
    return {"status": "accepted"}

@app.get("/test/webhook-received", response_model=List[dict])
def get_received_webhooks():
    return received_webhooks
