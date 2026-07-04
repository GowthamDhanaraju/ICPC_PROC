"""
Core job execution pipeline for the Batch Video Proctoring System.

Optimizations vs. original:
- Audio extraction and video frame sampling run concurrently via ThreadPoolExecutor.
- Webhook dispatch is non-blocking (runs in a daemon thread).
- Evidence frame lookup uses O(log N) binary search via find_nearest_frame().
- All print() replaced with structured logging.
- Whisper transcription removed: audio analysis now reports voice COUNT instead of text.
- Audio violation confidence is derived from the clustering score, not hardcoded.
- All storage is MinIO/S3 — per-job temp dirs are created and cleaned up automatically.
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
from app.database import SessionLocal, Job, Violation
from app.detection.audio_vad import detect_voice_activity
from app.detection.diarization import count_distinct_voices
from app.detection.face import detect_faces, verify_identity
from app.detection.gadget import detect_gadgets
from app.detection.gaze import estimate_gaze
from app.preprocessing.media import (
    extract_audio,
    find_nearest_frame,
    get_local_path,
    sample_video_frames,
    upload_evidence_frame,
)
from app.scoring.aggregator import EventAggregator
from app.scoring.scorer import ScoringEngine

logger = logging.getLogger(__name__)


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
      6. Aggregate events → calculate score.
      7. Upload evidence frames to MinIO/S3 (O(log N) lookup).
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

    # Create a job-scoped temp directory — ALL ephemeral files go here.
    # Cleaned up unconditionally in the finally block below.
    job_temp_dir = tempfile.mkdtemp(prefix=f"proctoring_{job_id[:8]}_")
    logger.debug(f"Job temp dir: {job_temp_dir}")

    try:
        local_video_path = get_local_path(job.source_video_s3_uri, job_temp_dir)
        logger.info(f"Video resolved: {local_video_path}")

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
        audio_violations: List[Dict[str, Any]] = []

        if audio_wav_path and os.path.exists(audio_wav_path):
            logger.info(f"Running VAD + Voice Count Diarization on {audio_wav_path}")
            speech_segments = detect_voice_activity(audio_wav_path)
            voice_result = count_distinct_voices(audio_wav_path, speech_segments)

            num_speakers    = voice_result["num_speakers"]
            flagged_segments = voice_result["flagged_segments"]
            diarization_confidence = voice_result["confidence"]

            if num_speakers > 1:
                logger.info(
                    f"Voice analysis: {num_speakers} distinct speaker(s) detected. "
                    f"{len(flagged_segments)} non-primary segment(s) flagged."
                )

            for start, end in flagged_segments:
                logger.info(
                    f"Non-primary voice at {start:.1f}s–{end:.1f}s "
                    f"(total speakers in session: {num_speakers})"
                )
                audio_violations.append({
                    "type": "SECOND_VOICE_DETECTED",
                    "start_ts": start,
                    "end_ts": end,
                    "duration": end - start,
                    "confidence": diarization_confidence,
                    "num_speakers": num_speakers,
                })
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
                    # Pass job_temp_dir so the enrollment photo is downloaded once
                    # and reused across all frames (natural per-job download cache)
                    sim = verify_identity(
                        face_crop,
                        job.enrollment_photo_s3_uri,
                        job.candidate_id,
                        job_temp_dir,
                    )
                    if sim < 0.6:
                        identity_mismatch_conf = 1.0 - sim

            gaze = {"yaw": 0.0, "pitch": 0.0}
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
        # Step 6: Aggregate → Score
        # ----------------------------------------------------------------
        aggregator = EventAggregator()
        raw_violations = aggregator.aggregate(frame_detections, audio_violations)

        # Roll up identity mismatch frames into a single violation span
        mismatch_frames = [fd for fd in frame_detections if "identity_mismatch" in fd]
        if mismatch_frames:
            start_m  = mismatch_frames[0]["timestamp"]
            end_m    = mismatch_frames[-1]["timestamp"]
            avg_conf = sum(fd["identity_mismatch"] for fd in mismatch_frames) / len(mismatch_frames)
            raw_violations.append({
                "type": "IDENTITY_MISMATCH",
                "start_ts": start_m,
                "end_ts": end_m,
                "duration": max(end_m - start_m, settings.ADAPTIVE_SAMPLING_BASELINE_INTERVAL),
                "confidence": avg_conf,
            })

        raw_violations.sort(key=lambda v: v["start_ts"])

        scorer = ScoringEngine()
        overall_score = scorer.calculate_score(raw_violations)

        # ----------------------------------------------------------------
        # Step 7: Persist violations + evidence frames (O(log N) lookup)
        # ----------------------------------------------------------------
        for violation in raw_violations:
            closest_frame = find_nearest_frame(frames_list, violation["start_ts"])
            evidence_uri = None

            if closest_frame is not None:
                ok, encoded = cv2.imencode(".jpg", closest_frame)
                if ok:
                    evidence_uri = upload_evidence_frame(
                        job.id, violation["start_ts"], encoded.tobytes()
                    )

            db.add(Violation(
                job_id=job.id,
                type=violation["type"],
                start_ts=violation["start_ts"],
                end_ts=violation["end_ts"],
                duration=violation["duration"],
                confidence=violation["confidence"],
                evidence_frame_s3_uri=evidence_uri,
            ))

        # ----------------------------------------------------------------
        # Step 8: Finalize job
        # ----------------------------------------------------------------
        job.overall_score = overall_score
        job.status = "COMPLETED"
        job.error_message = None
        db.commit()
        db.refresh(job)
        logger.info(f"Job {job_id} COMPLETED. Score: {overall_score:.2f}/100")

        _dispatch_webhook_async(job)

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
        # Always clean up all ephemeral processing files for this job
        shutil.rmtree(job_temp_dir, ignore_errors=True)
        logger.debug(f"Temp dir cleaned up: {job_temp_dir}")


# ---------------------------------------------------------------------------
# Webhook Dispatch (non-blocking)
# ---------------------------------------------------------------------------

def _dispatch_webhook_async(job: Job):
    """
    Fires webhook delivery in a daemon thread so retries don't block the worker.
    """
    if not job.webhook_url:
        return
    payload = {
        "job_id": job.id,
        "candidate_id": job.candidate_id,
        "status": job.status,
        "overall_score": job.overall_score if job.status == "COMPLETED" else None,
        "error_message": job.error_message if job.status == "FAILED" else None,
        "violations": [v.to_dict() for v in job.violations],
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
