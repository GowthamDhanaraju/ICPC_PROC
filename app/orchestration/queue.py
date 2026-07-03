"""
Job queue management for the Batch Video Proctoring Pipeline.

Provides two dispatch mechanisms:
  1. JobQueueManager — lightweight in-process thread queue (dev / single-process).
  2. SQSListener     — AWS SQS long-poll consumer for S3 ObjectCreated events.

In production, prefer Celery (app/orchestration/tasks.py) over JobQueueManager
for durable, restartable job execution.
"""
import json
import logging
import queue
import threading
import time
import urllib.parse

import boto3

from app.config import settings
from app.database import SessionLocal, Job
from app.orchestration.worker import process_job

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-process Thread Queue (development / single-worker fallback)
# ---------------------------------------------------------------------------

class JobQueueManager:
    """
    Simple thread-pool backed in-memory queue.
    WARNING: Jobs in this queue are lost if the process restarts.
    Use Celery (tasks.py) for production deployments.
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._workers: list = []
        self._running = False

    def submit(self, job_id: str):
        logger.info(f"Queuing job {job_id} for in-process worker.")
        self._queue.put(job_id)

    def start(self, num_workers: int = 2):
        if self._running:
            return
        self._running = True
        for i in range(num_workers):
            t = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            t.start()
            self._workers.append(t)
        logger.info(f"Started {num_workers} in-process worker threads.")

    def stop(self):
        self._running = False
        for _ in self._workers:
            self._queue.put(None)  # Unblock blocked get() calls
        for t in self._workers:
            t.join(timeout=2.0)
        self._workers = []
        logger.info("In-process worker threads stopped.")

    def _worker_loop(self, worker_id: int):
        logger.info(f"Worker-{worker_id} started.")
        while self._running:
            try:
                job_id = self._queue.get(timeout=1.0)
                if job_id is None:
                    break
                logger.info(f"Worker-{worker_id} processing job: {job_id}")
                try:
                    process_job(job_id)
                except Exception as exc:
                    logger.error(f"Worker-{worker_id} error on job {job_id}: {exc}")
                finally:
                    self._queue.task_done()
            except queue.Empty:
                continue
        logger.info(f"Worker-{worker_id} exiting.")


# Global singleton — used by the FastAPI app in single-process mode
global_queue = JobQueueManager()


# ---------------------------------------------------------------------------
# SQS Listener (event-driven S3 trigger mode)
# ---------------------------------------------------------------------------

class SQSListener:
    """
    Long-polls an SQS queue for S3 ObjectCreated notifications,
    creates proctoring jobs, and dispatches them for processing.
    """

    def __init__(self, queue_url: str):
        self.queue_url = queue_url
        self.sqs = boto3.client("sqs", region_name=settings.AWS_REGION)
        self.running = False

    def start(self):
        self.running = True
        t = threading.Thread(target=self._poll_loop, daemon=True)
        t.start()
        logger.info(f"SQS Listener polling: {self.queue_url}")

    def stop(self):
        self.running = False

    def _poll_loop(self):
        while self.running:
            try:
                response = self.sqs.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=5,
                    WaitTimeSeconds=10,  # Long polling
                )
                for msg in response.get("Messages", []):
                    try:
                        body = json.loads(msg["Body"])
                        if "Records" in body:
                            for record in body["Records"]:
                                s3_info = record.get("s3", {})
                                bucket = s3_info.get("bucket", {}).get("name")
                                key = urllib.parse.unquote_plus(
                                    s3_info.get("object", {}).get("key", "")
                                )
                                if bucket and key:
                                    self._create_and_submit_job(f"s3://{bucket}/{key}")
                    except Exception as exc:
                        logger.error(f"Failed to process SQS message: {exc}")
                    finally:
                        self.sqs.delete_message(
                            QueueUrl=self.queue_url,
                            ReceiptHandle=msg["ReceiptHandle"],
                        )
            except Exception as exc:
                logger.error(f"SQS polling error: {exc}")
                time.sleep(5)

    def _create_and_submit_job(self, source_uri: str):
        db = SessionLocal()
        try:
            # Idempotency: skip if an active job already exists for this URI
            existing = db.query(Job).filter(
                Job.source_video_s3_uri == source_uri,
                Job.status.in_(["QUEUED", "PROCESSING"]),
            ).first()
            if existing:
                logger.info(
                    f"Active job {existing.id} already exists for {source_uri}. Skipping."
                )
                return

            # Extract candidate ID from key path: bucket/candidate_id/session.mp4
            parts = source_uri.split("/")
            candidate_id = parts[-2] if len(parts) >= 4 else "unknown"

            job = Job(
                candidate_id=candidate_id,
                source_video_s3_uri=source_uri,
                status="QUEUED",
            )
            db.add(job)
            db.commit()
            db.refresh(job)
            logger.info(f"Auto-created job {job.id} from SQS event: {source_uri}")
            global_queue.submit(job.id)
        except Exception as exc:
            logger.error(f"Failed to auto-create job from SQS event {source_uri}: {exc}")
        finally:
            db.close()
