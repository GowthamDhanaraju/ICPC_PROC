import os
import cv2
import time
import requests
import traceback
from typing import List, Dict, Any, Tuple
from app.config import settings
from app.database import SessionLocal, Job, Violation
from app.preprocessing.media import get_local_path, extract_audio, sample_video_frames, upload_evidence_frame
from app.detection.face import detect_faces, verify_identity
from app.detection.gaze import estimate_gaze
from app.detection.gadget import detect_gadgets
from app.detection.audio_vad import detect_voice_activity
from app.detection.diarization import detect_multiple_speakers
from app.detection.whisper_transcription import transcribe_segment
from app.scoring.aggregator import EventAggregator
from app.scoring.scorer import ScoringEngine

def process_job(job_id: str):
    """
    Main job executor pipeline:
    1. Fetch job from DB and mark PROCESSING.
    2. Retrieve video file path.
    3. Extract audio and run VAD, Diarization, and Whisper.
    4. Run adaptive sampling and visual models on frames.
    5. Aggregate events and calculate score.
    6. Extract evidence frames and write violations.
    7. Commit updates and send webhook callback.
    """
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        print(f"Error: Job {job_id} not found in database.")
        db.close()
        return

    # Update state to PROCESSING
    job.status = "PROCESSING"
    db.commit()
    db.refresh(job)
    print(f"Job {job_id} status updated to PROCESSING. Starting pipeline...")

    try:
        # Step 2: Download / resolve local path
        local_video_path = get_local_path(job.source_video_s3_uri)
        print(f"Using local video source path: {local_video_path}")

        # Step 3: Audio analysis (Concurrent/Separate stream)
        audio_violations: List[Dict[str, Any]] = []
        audio_wav_path = extract_audio(local_video_path, job.id)
        
        if audio_wav_path and os.path.exists(audio_wav_path):
            print(f"Extracted audio WAV: {audio_wav_path}. Running VAD and Diarization...")
            speech_segments = detect_voice_activity(audio_wav_path)
            multi_speaker_segments = detect_multiple_speakers(audio_wav_path, speech_segments)
            
            for start, end in multi_speaker_segments:
                # Targeted Whisper transcription on suspected speech
                transcript = transcribe_segment(audio_wav_path, start, end)
                print(f"Suspicious audio segment ({start:.1f}s - {end:.1f}s) transcription: '{transcript}'")
                
                # Check for suspicious transcript content in production if needed
                audio_violations.append({
                    "type": "SECOND_VOICE_DETECTED",
                    "start_ts": start,
                    "end_ts": end,
                    "duration": end - start,
                    "confidence": 0.88,
                    # We can store the transcript inside local db logs/metadata
                })

        # Step 4: Video analysis via Adaptive Frame Sampling
        sampled_frames = sample_video_frames(local_video_path)
        frame_detections: List[Dict[str, Any]] = []
        
        print("Running visual detection models on sampled frames...")
        for ts, frame_img in sampled_frames:
            # 1. Face Count
            faces = detect_faces(frame_img, ts)
            
            # 2. Identity Verification (only if enrollment photo exists and face is detected)
            identity_mismatch_detected = False
            identity_confidence = 1.0
            
            if job.enrollment_photo_s3_uri and len(faces) > 0:
                # Crop first face for verification
                x1, y1, x2, y2 = faces[0]["box"]
                face_crop = frame_img[max(0, y1):y2, max(0, x1):x2]
                if face_crop.size > 0:
                    sim_score = verify_identity(face_crop, job.enrollment_photo_s3_uri, job.candidate_id)
                    if sim_score < 0.6:  # Identity mismatch threshold
                        identity_mismatch_detected = True
                        identity_confidence = 1.0 - sim_score

            # 3. Gaze / Head Pose (only if face is detected)
            gaze = {"yaw": 0.0, "pitch": 0.0}
            if len(faces) > 0:
                gaze = estimate_gaze(frame_img, faces[0]["box"], ts)

            # 4. Prohibited Device detection
            gadgets = detect_gadgets(frame_img, ts)

            # Assemble frame details
            fd = {
                "timestamp": ts,
                "faces": faces,
                "gaze": gaze,
                "gadgets": gadgets
            }
            
            # Inject identity mismatch as custom detection type to feed aggregator
            if identity_mismatch_detected:
                # Add fake faces if we need to force aggregator, or we can just append it directly as a flag
                # Let's map it into fd for custom aggregator check
                fd["identity_mismatch"] = identity_confidence

            frame_detections.append(fd)

        # Step 5: Event Aggregation
        aggregator = EventAggregator()
        raw_violations = aggregator.aggregate(frame_detections, audio_violations)
        
        # Inject identity mismatch explicitly if detected
        # If any frame had identity mismatch, group them into IDENTITY_MISMATCH violation
        mismatch_frames = [fd for fd in frame_detections if "identity_mismatch" in fd]
        if mismatch_frames:
            start_m = mismatch_frames[0]["timestamp"]
            end_m = mismatch_frames[-1]["timestamp"]
            duration_m = end_m - start_m
            avg_m_conf = sum(fd["identity_mismatch"] for fd in mismatch_frames) / len(mismatch_frames)
            
            raw_violations.append({
                "type": "IDENTITY_MISMATCH",
                "start_ts": start_m,
                "end_ts": end_m,
                "duration": max(duration_m, settings.ADAPTIVE_SAMPLING_BASELINE_INTERVAL),
                "confidence": avg_m_conf
            })

        # Sort violations by time
        raw_violations.sort(key=lambda x: x["start_ts"])

        # Step 6: Extract evidence frames and write to database
        for violation in raw_violations:
            # Find closest sampled frame to capture as evidence image
            evidence_uri = None
            closest_frame_img = None
            min_diff = float("inf")
            
            for ts, frame_img in sampled_frames:
                diff = abs(ts - violation["start_ts"])
                if diff < min_diff:
                    min_diff = diff
                    closest_frame_img = frame_img
            
            if closest_frame_img is not None:
                # Encode frame to JPG bytes
                success, encoded_img = cv2.imencode(".jpg", closest_frame_img)
                if success:
                    # Save/upload JPG
                    evidence_uri = upload_evidence_frame(job.id, violation["start_ts"], encoded_img.tobytes())
            
            # Save violation in DB
            db_violation = Violation(
                job_id=job.id,
                type=violation["type"],
                start_ts=violation["start_ts"],
                end_ts=violation["end_ts"],
                duration=violation["duration"],
                confidence=violation["confidence"],
                evidence_frame_s3_uri=evidence_uri
            )
            db.add(db_violation)

        # Step 7: Scoring Engine
        scorer = ScoringEngine()
        overall_score = scorer.calculate_score(raw_violations)

        # Finalize job parameters
        job.overall_score = overall_score
        job.status = "COMPLETED"
        job.error_message = None
        db.commit()
        db.refresh(job)
        print(f"Job {job_id} successfully completed. Score: {overall_score}")

        # Send Webhook callback
        send_webhook(job)

    except Exception as e:
        # Handle failure
        db.rollback()
        err_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Pipeline error processing job {job_id}: {err_msg}")
        
        job.status = "FAILED"
        job.error_message = err_msg
        db.commit()
        
        # Send failure callback
        send_webhook(job)

    finally:
        db.close()

def send_webhook(job: Job):
    """
    Dispatches HTTP POST webhook notifications on job completion or failure.
    """
    if not job.webhook_url:
        print(f"No webhook URL configured for job {job.id}. Skipping callback.")
        return

    payload = {
        "job_id": job.id,
        "candidate_id": job.candidate_id,
        "status": job.status,
        "overall_score": job.overall_score if job.status == "COMPLETED" else None,
        "error_message": job.error_message if job.status == "FAILED" else None,
        "violations": [v.to_dict() for v in job.violations]
    }

    max_retries = settings.WEBHOOK_MAX_RETRIES
    timeout = settings.WEBHOOK_TIMEOUT

    print(f"Sending webhook callback to {job.webhook_url}...")
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(job.webhook_url, json=payload, timeout=timeout)
            if response.status_code in (200, 201, 202, 204):
                print(f"Webhook delivered successfully on attempt {attempt}.")
                return
            else:
                print(f"Webhook returned status code {response.status_code} on attempt {attempt}.")
        except Exception as e:
            print(f"Webhook delivery failed on attempt {attempt}: {e}")
        
        if attempt < max_retries:
            time.sleep(2 ** attempt)  # Exponential backoff

    print(f"Error: Webhook delivery failed after {max_retries} attempts.")
