"""
FastAPI application entrypoint for the Batch Video Proctoring Pipeline.

Key changes vs. original:
- Uses @asynccontextmanager lifespan instead of deprecated @app.on_event("startup").
- /health and /ready endpoints for Kubernetes / ECS health probes.
- Test routes (/test/*) are only mounted when TESTING_MODE=True.
- Pydantic schemas imported from app.schemas (not inline).
- Request logging middleware.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db, init_db, Job, engine
from app.logging_config import configure_logging
from app.orchestration.queue import global_queue
from app.schemas import (
    HealthResponse,
    JobCreate,
    JobResponse,
    JobSubmitResponse,
    ReadyResponse,
)

configure_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle manager."""
    logger.info("=== Proctoring API starting up ===")
    init_db()
    global_queue.start(num_workers=settings.WORKER_CONCURRENCY)
    logger.info(f"Worker queue started with {settings.WORKER_CONCURRENCY} threads.")
    yield
    logger.info("=== Proctoring API shutting down ===")
    global_queue.stop()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Batch Video Proctoring API",
    description=(
        "S3 event-driven / REST batch proctoring pipeline. "
        "Submit a video S3 URI, poll for results, receive webhook callbacks."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Lightweight request logger — logs method, path, and response status."""
    response = await call_next(request)
    # Skip high-frequency health check noise
    if request.url.path not in ("/health", "/ready"):
        logger.info(f"{request.method} {request.url.path} → {response.status_code}")
    return response


# ---------------------------------------------------------------------------
# Health & Readiness Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health():
    """
    Liveness probe. Returns 200 if the process is running.
    Used by Kubernetes / ECS to detect crash-looping containers.
    """
    return HealthResponse(status="ok", version=app.version)


@app.get("/ready", response_model=ReadyResponse, tags=["Ops"])
def ready(db: Session = Depends(get_db)):
    """
    Readiness probe. Checks that the database is reachable.
    Returns 503 if any dependency is unavailable.
    """
    db_status = "ok"
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception as exc:
        logger.error(f"DB readiness check failed: {exc}")
        db_status = "unavailable"

    queue_status = "ok" if global_queue._running else "stopped"

    if db_status != "ok":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "database": db_status, "queue": queue_status},
        )

    return ReadyResponse(status="ready", database=db_status, queue=queue_status)


# ---------------------------------------------------------------------------
# Proctoring API Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/v1/sessions",
    response_model=JobSubmitResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Proctoring"],
)
def submit_session(payload: JobCreate, db: Session = Depends(get_db)):
    """
    Submits a batch video proctoring job.
    Inserts a job record into the database and queues it for background processing.
    Idempotent: returns the existing job if one is already active for the same video + candidate.
    """
    try:
        existing = db.query(Job).filter(
            Job.source_video_s3_uri == payload.video_s3_uri,
            Job.candidate_id == payload.candidate_id,
            Job.status.in_(["QUEUED", "PROCESSING"]),
        ).first()

        if existing:
            return JobSubmitResponse(
                job_id=existing.id,
                status=existing.status,
                message="Active job already exists for this video and candidate.",
            )

        job = Job(
            candidate_id=payload.candidate_id,
            source_video_s3_uri=payload.video_s3_uri,
            enrollment_photo_s3_uri=payload.enrollment_photo_s3_uri,
            webhook_url=payload.webhook_url,
            status="QUEUED",
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        # Dispatch job based on environment
        if settings.ENV == "production" and settings.CELERY_BROKER_URL:
            from app.orchestration.tasks import process_job_task
            process_job_task.delay(job.id)
            logger.info(f"Job {job.id} dispatched to Celery for candidate {job.candidate_id}.")
        else:
            global_queue.submit(job.id)
            logger.info(f"Job {job.id} submitted to local thread queue for candidate {job.candidate_id}.")

        return JobSubmitResponse(job_id=job.id, status=job.status)

    except Exception as exc:
        db.rollback()
        logger.error(f"Failed to submit proctoring job: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit proctoring job: {exc}",
        )


@app.get(
    "/v1/sessions/{job_id}",
    response_model=JobResponse,
    tags=["Proctoring"],
)
def get_session(job_id: str, db: Session = Depends(get_db)):
    """
    Returns the current status, score, and violation timeline for a proctoring job.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )
    return job.to_dict()


# ---------------------------------------------------------------------------
# Test Routes (only mounted in TESTING_MODE)
# ---------------------------------------------------------------------------

if settings.TESTING_MODE:
    from fastapi import APIRouter
    from typing import List

    _received_webhooks: List[dict] = []
    test_router = APIRouter(prefix="/test", tags=["Testing"])

    @test_router.post("/webhook-target", status_code=200)
    def test_webhook_target(payload: dict):
        logger.info(
            f"[TEST WEBHOOK] job={payload.get('job_id')} "
            f"status={payload.get('status')} score={payload.get('overall_score')}"
        )
        _received_webhooks.append(payload)
        return {"status": "accepted"}

    @test_router.get("/webhook-received")
    def get_received_webhooks():
        return _received_webhooks

    app.include_router(test_router)
    logger.warning("TESTING_MODE=True — test routes are mounted. Do NOT use in production.")
