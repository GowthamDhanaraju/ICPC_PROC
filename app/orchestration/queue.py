import queue
import threading
import time
import json
import urllib.parse
import boto3
from typing import Callable, Optional
from app.config import settings
from app.database import SessionLocal, Job
from app.orchestration.worker import process_job

class JobQueueManager:
    def __init__(self):
        self._queue = queue.Queue()
        self._workers = []
        self._running = False

    def submit(self, job_id: str):
        """Submit a job ID to the execution queue."""
        print(f"Submitting job {job_id} to internal worker queue")
        self._queue.put(job_id)

    def start(self, num_workers: int = 2):
        """Starts worker threads to process queue items."""
        if self._running:
            return
        self._running = True
        for i in range(num_workers):
            t = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            t.start()
            self._workers.append(t)
        print(f"Started {num_workers} background worker threads.")

    def stop(self):
        """Stops worker threads."""
        self._running = False
        # Put None to unblock queue get
        for _ in self._workers:
            self._queue.put(None)
        for t in self._workers:
            t.join(timeout=1.0)
        self._workers = []

    def _worker_loop(self, worker_id: int):
        while self._running:
            try:
                job_id = self._queue.get(timeout=1.0)
                if job_id is None:
                    break
                print(f"Worker-{worker_id} started processing job: {job_id}")
                try:
                    process_job(job_id)
                except Exception as e:
                    print(f"Worker-{worker_id} error processing job {job_id}: {e}")
                finally:
                    self._queue.task_done()
            except queue.Empty:
                continue

# Singleton queue manager
global_queue = JobQueueManager()


class SQSListener:
    """
    Listens to an SQS Queue for S3 ObjectCreated event notifications,
    creates jobs in the database, and schedules them for processing.
    """
    def __init__(self, queue_url: str):
        self.queue_url = queue_url
        self.sqs = boto3.client('sqs', region_name=settings.AWS_REGION)
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._poll_loop, daemon=True).start()
        print(f"SQS Listener started polling on: {self.queue_url}")

    def stop(self):
        self.running = False

    def _poll_loop(self):
        while self.running:
            try:
                response = self.sqs.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=5,
                    WaitTimeSeconds=10  # Long polling
                )
                
                messages = response.get('Messages', [])
                for msg in messages:
                    body = json.loads(msg['Body'])
                    
                    # SQS messages from S3 event notifications contain a Records list
                    if "Records" in body:
                        for record in body["Records"]:
                            s3 = record.get("s3", {})
                            bucket = s3.get("bucket", {}).get("name")
                            key = urllib.parse.unquote_plus(s3.get("object", {}).get("key"))
                            etag = s3.get("object", {}).get("eTag")
                            
                            if bucket and key:
                                s3_uri = f"s3://{bucket}/{key}"
                                self._create_and_submit_job(s3_uri)
                    
                    # Delete message from SQS queue
                    self.sqs.delete_message(
                        QueueUrl=self.queue_url,
                        ReceiptHandle=msg['ReceiptHandle']
                    )
            except Exception as e:
                print(f"SQS polling error: {e}")
                time.sleep(5)  # Backoff

    def _create_and_submit_job(self, source_uri: str):
        db = SessionLocal()
        try:
            # Check idempotency (prevent double processing of same key)
            existing = db.query(Job).filter(Job.source_video_s3_uri == source_uri).first()
            if existing:
                print(f"S3 object {source_uri} already registered in database (job {existing.id}). Skipping.")
                return

            # Extract candidate ID from key if possible, e.g. "incoming/candidate123/session.mp4"
            parts = source_uri.split('/')
            candidate_id = "unknown"
            if len(parts) >= 4:
                candidate_id = parts[-2]

            job = Job(
                candidate_id=candidate_id,
                source_video_s3_uri=source_uri,
                status="QUEUED"
            )
            db.add(job)
            db.commit()
            db.refresh(job)
            
            # Submit to processing queue
            global_queue.submit(job.id)
        except Exception as e:
            print(f"Failed to auto-create job from SQS event: {e}")
        finally:
            db.close()
