"""
Celery task definitions for the Batch Video Proctoring Pipeline.

Key design choices:
- acks_late=True: The SQS/Redis message is only acknowledged AFTER the task
  completes. If the worker crashes mid-job, the broker re-delivers the task.
- max_retries=3: Transient failures (e.g., S3 timeout) are automatically retried
  with exponential backoff before marking a job FAILED.
- bind=True: The task instance is passed as `self`, enabling `self.retry()`.
"""
import logging
from celery import Celery
from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Celery application
# ---------------------------------------------------------------------------
celery_app = Celery(
    "proctoring_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    # Task serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Reliability
    task_acks_late=True,           # Acknowledge only after completion
    task_reject_on_worker_lost=True,  # Re-queue if worker process is killed
    worker_prefetch_multiplier=1,  # One task per worker at a time (CPU-bound)
    # Retry config
    task_soft_time_limit=3600,     # 1 hour soft limit (raises SoftTimeLimitExceeded)
    task_time_limit=3900,          # 1 hour 5 min hard kill limit
    # Result expiry
    result_expires=86400,          # Keep task results for 24 hours
    # Timezone
    timezone="UTC",
    enable_utc=True,
)


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------
@celery_app.task(
    bind=True,
    max_retries=3,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
)
def process_job_task(self, job_id: str):
    """
    Celery task entrypoint for processing a video proctoring session.
    Automatically retries on transient exceptions with exponential backoff.
    """
    from app.orchestration.worker import process_job
    try:
        logger.info(f"Celery worker started job {job_id}")
        process_job(job_id)
        logger.info(f"Celery worker completed job {job_id}")
    except Exception as exc:
        logger.error(f"Celery worker failed job {job_id}: {exc}")
        raise exc
