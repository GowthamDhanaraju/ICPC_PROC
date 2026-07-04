"""
Core job execution pipeline for the Batch Video Proctoring System.

Changes from original:
- Scoring system (ScoringEngine, EventAggregator, overall_score, violations DB table)
  has been removed entirely.
- The pipeline now calls build_report() to produce a structured JSON observation log
  and saves it to reports/<job_id>.json.
- Audio extraction and video frame sampling still run concurrently.
- Webhook payload updated: contains report_path instead of score/violations.
"""
import logging
import os
import shutil
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import cv2
import requests

from app.config import settings
from app.database import SessionLocal, Job
from app.detection.audio_vad import detect_voice_activity
from app.detection.diarization import count_distinct_voices
from app.detection.face import detect_faces, verify_identity
from app.detection.gadget import detect_gadgets
from app.detection.gaze import estimate_gaze
from app.preprocessing.media import (
    extract_audio,
    get_local_path,
    sample_video_frames,
)
from app.reporting.report import build_report, save_report

logger = logging.getLogger(__name__)

# Directory where JSON reports are written (relative to CWD / project root)
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _video_duration(video_path: str) -> Optional[float]:
    """Returns the duration of a video file in seconds using OpenCV."""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps > 0:
            return frame_count / fps
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def process_job(job_id: str):
    """
    Full proctoring pipeline:
      1. Fetch job & mark PROCESSING.
      2. Create a per-job temp directory for all ephemeral files.
      3. Audio extraction + Video frame sampling — run concurrently.
      4. Run VAD → Voice Count Diarization on audio.
      5. Run face / gaze / gadget detection on sampled frames.
      6. Build structured JSON report from all detection outputs.
      7. Save report to disk; store path on job record.
      8. Commit results, fire webhook asynchronously.
      9. Clean up the per-job temp directory (always runs, even on failure).
    """
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        logger.error(f"Job {job_id} not found in database.")
        db.close()
        return

    job.status = "PROCESSING"
    db.commit()
    db.refresh(job)
    logger.info(f"Job {job_id} → PROCESSING")

    job_temp_dir = tempfile.mkdtemp(prefix=f"proctoring_{job_id[:8]}_")
    logger.debug(f"Job temp dir: {job_temp_dir}")

    try:
        local_video_path = get_local_path(job.source_video_s3_uri, job_temp_dir)
        logger.info(f"Video resolved: {local_video_path}")

        # Get video duration for report metadata
        duration_s = _video_duration(local_video_path)

        # ----------------------------------------------------------------
        # Step 3: Concurrently extract audio + sample video frames
        # ----------------------------------------------------------------
        frames_list: List = []
        audio_wav_path: Optional[str] = None

        with ThreadPoolExecutor(max_workers=2) as pool:
            future_audio = pool.submit(extract_audio, local_video_path, job.id, job_temp_dir)
            future_video = pool.submit(sample_video_frames, local_video_path)

            for fut in as_completed([future_audio, future_video]):
                if fut is future_audio:
                    audio_wav_path = fut.result()
                else:
                    frames_list, _ = fut.result()

        # ----------------------------------------------------------------
        # Step 4: Audio analysis — VAD + Voice Count Diarization
        # ----------------------------------------------------------------
        has_audio = bool(audio_wav_path and os.path.exists(audio_wav_path))
        speech_segments: List = []
        voice_result: Optional[Dict[str, Any]] = None

        if has_audio:
            logger.info(f"Running VAD + Voice Count Diarization on {audio_wav_path}")
            speech_segments = detect_voice_activity(audio_wav_path)
            voice_result = count_distinct_voices(audio_wav_path, speech_segments)

            num_speakers = voice_result["num_speakers"]
            flagged_segments = voice_result["flagged_segments"]
            logger.info(
                f"Voice analysis: {num_speakers} distinct speaker(s) detected. "
                f"{len(flagged_segments)} non-primary segment(s) flagged."
            )
        else:
            logger.info("No audio track found — skipping audio analysis.")

        # ----------------------------------------------------------------
        # Step 5: Visual detection on sampled frames
        # ----------------------------------------------------------------
        logger.info(f"Running visual detection on {len(frames_list)} frames...")
        frame_detections: List[Dict[str, Any]] = []

        for ts, frame_img in frames_list:
            faces = detect_faces(frame_img, ts)

            identity_mismatch_conf = None
            if job.enrollment_photo_s3_uri and faces:
                x1, y1, x2, y2 = faces[0]["box"]
                face_crop = frame_img[max(0, y1):y2, max(0, x1):x2]
                if face_crop.size > 0:
                    sim = verify_identity(
                        face_crop,
                        job.enrollment_photo_s3_uri,
                        job.candidate_id,
                        job_temp_dir,
                    )
                    if sim < 0.6:
                        identity_mismatch_conf = 1.0 - sim

            gaze: Dict[str, float] = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
            if faces:
                gaze = estimate_gaze(frame_img, faces[0]["box"], ts)

            gadgets = detect_gadgets(frame_img, ts)

            fd: Dict[str, Any] = {
                "timestamp": ts,
                "faces": faces,
                "gaze": gaze,
                "gadgets": gadgets,
            }
            if identity_mismatch_conf is not None:
                fd["identity_mismatch"] = identity_mismatch_conf

            frame_detections.append(fd)

        # ----------------------------------------------------------------
        # Step 6: Build JSON report
        # ----------------------------------------------------------------
        logger.info("Building JSON detection report...")
        report = build_report(
            job_id=job.id,
            video_path=local_video_path,
            candidate_id=job.candidate_id,
            frame_detections=frame_detections,
            audio_speech_segments=speech_segments,
            voice_result=voice_result,
            has_audio=has_audio,
            video_duration_s=duration_s,
            frame_count=len(frames_list),
        )

        # ----------------------------------------------------------------
        # Step 7: Save report to disk
        # ----------------------------------------------------------------
        report_path = os.path.join(REPORTS_DIR, f"{job.id}.json")
        saved_path = save_report(report, report_path)
        logger.info(f"Report saved → {saved_path}")

        # ----------------------------------------------------------------
        # Step 8: Finalize job
        # ----------------------------------------------------------------
        job.status = "COMPLETED"
        job.error_message = None
        # Store report path on the job if the column exists
        if hasattr(job, "report_path"):
            job.report_path = saved_path
        db.commit()
        db.refresh(job)
        logger.info(f"Job {job_id} COMPLETED. Report: {saved_path}")

        _dispatch_webhook_async(job, report_path=saved_path)

    except Exception as exc:
        db.rollback()
        err_msg = f"{exc}\n{traceback.format_exc()}"
        logger.error(f"Pipeline error for job {job_id}: {err_msg}")

        job.status = "FAILED"
        job.error_message = err_msg[:4000]
        db.commit()
        _dispatch_webhook_async(job)

    finally:
        db.close()
        shutil.rmtree(job_temp_dir, ignore_errors=True)
        logger.debug(f"Temp dir cleaned up: {job_temp_dir}")


# ---------------------------------------------------------------------------
# Webhook Dispatch (non-blocking)
# ---------------------------------------------------------------------------

def _dispatch_webhook_async(job: Job, report_path: Optional[str] = None):
    """
    Fires webhook delivery in a daemon thread so retries don't block the worker.
    """
    if not job.webhook_url:
        return
    payload = {
        "job_id": job.id,
        "candidate_id": job.candidate_id,
        "status": job.status,
        "report_path": report_path if job.status == "COMPLETED" else None,
        "error_message": job.error_message if job.status == "FAILED" else None,
    }
    webhook_url = job.webhook_url
    t = threading.Thread(
        target=_send_webhook,
        args=(webhook_url, payload),
        daemon=True,
    )
    t.start()


def _send_webhook(webhook_url: str, payload: dict):
    """
    Delivers a webhook with exponential-backoff retries.
    Runs in a daemon thread — does NOT block the calling worker.
    """
    max_retries = settings.WEBHOOK_MAX_RETRIES
    timeout     = settings.WEBHOOK_TIMEOUT

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(webhook_url, json=payload, timeout=timeout)
            if resp.status_code in (200, 201, 202, 204):
                logger.info(f"Webhook delivered to {webhook_url} on attempt {attempt}.")
                return
            logger.warning(
                f"Webhook attempt {attempt} returned HTTP {resp.status_code}."
            )
        except Exception as exc:
            logger.warning(f"Webhook attempt {attempt} failed: {exc}")

        if attempt < max_retries:
            time.sleep(2 ** attempt)  # 2s, 4s, 8s

    logger.error(f"Webhook delivery to {webhook_url} failed after {max_retries} attempts.")
