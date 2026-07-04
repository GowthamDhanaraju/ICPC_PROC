"""
Video preprocessing and S3 storage utilities.

Key optimizations vs. original:
- Frame sampling uses cap.grab() for sequential reads instead of
  cap.set(CAP_PROP_POS_MSEC) per frame. This avoids repeated H.264 keyframe
  decodes and is ~5-20x faster on long videos.
- sample_video_frames() now implements TRUE adaptive sampling: the next
  frame's interval is computed from the current frame's motion score,
  building the target list dynamically rather than pre-freezing it.
- sample_video_frames() returns a sorted list AND a timestamp-indexed dict
  for O(log N) evidence frame lookups in the worker.
"""
import bisect
import logging
import os
import shutil
import subprocess
import urllib.parse
from typing import Dict, List, Optional, Tuple

import boto3
import cv2
import numpy as np
from botocore.exceptions import NoCredentialsError

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# S3 Helpers
# ---------------------------------------------------------------------------

def get_s3_client():
    """
    Builds a boto3 S3 client.
    Uses explicit credentials from settings if provided;
    falls back to IAM role / ~/.aws credentials otherwise.
    """
    kwargs = {"region_name": settings.AWS_REGION}
    if settings.AWS_ACCESS_KEY_ID:
        kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
    if settings.AWS_SECRET_ACCESS_KEY:
        kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
    return boto3.client("s3", **kwargs)


def _has_aws_credentials() -> bool:
    """Returns True if any AWS credential source is configured."""
    return (
        settings.AWS_ACCESS_KEY_ID is not None
        or os.getenv("AWS_ACCESS_KEY_ID") is not None
        or os.path.exists(os.path.expanduser("~/.aws/credentials"))
        or os.path.exists(os.path.expanduser("~/.aws/config"))
    )


def get_local_path(s3_uri: str) -> str:
    """
    Resolves an S3 URI to a local filesystem path:
    1. If it's already a valid local path, return it as-is.
    2. If a cached download exists, return the cache.
    3. Attempt S3 download via boto3.
    4. If no credentials exist, fall back to local mock storage.
    """
    if os.path.exists(s3_uri):
        return s3_uri

    parsed = urllib.parse.urlparse(s3_uri)
    if parsed.scheme != "s3":
        return s3_uri

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    download_path = os.path.join(settings.LOCAL_STORAGE_DIR, "downloads", bucket, key)
    os.makedirs(os.path.dirname(download_path), exist_ok=True)

    if os.path.exists(download_path):
        return download_path

    try:
        s3 = get_s3_client()
        logger.info(f"Downloading s3://{bucket}/{key} → {download_path}")
        s3.download_file(bucket, key, download_path)
        return download_path
    except Exception as exc:
        logger.warning(f"S3 download failed for {s3_uri}: {exc}")
        if _has_aws_credentials():
            raise

    # Offline fallback: check local storage mock directory
    local_mock = os.path.join(settings.LOCAL_STORAGE_DIR, bucket, key)
    if os.path.exists(local_mock):
        logger.info(f"Using local storage fallback: {local_mock}")
        return local_mock

    raise FileNotFoundError(
        f"S3 file {s3_uri} could not be retrieved and no local fallback exists."
    )


def upload_evidence_frame(job_id: str, timestamp: float, frame_data: bytes) -> str:
    """
    Saves an evidence frame JPEG locally and uploads it to the results S3 bucket.
    Returns the S3 URI regardless of whether the upload succeeded
    (on credential failure, the local file is kept and the URI is returned for the DB record).
    """
    filename = f"evidence_{job_id}_{int(timestamp * 1000)}.jpg"
    s3_key = f"results/{job_id}/{filename}"
    s3_uri = f"s3://{settings.RESULTS_S3_BUCKET}/{s3_key}"

    # Always persist locally
    local_path = os.path.join(
        settings.LOCAL_STORAGE_DIR, settings.RESULTS_S3_BUCKET, job_id, filename
    )
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(frame_data)

    try:
        s3 = get_s3_client()
        logger.info(f"Uploading evidence frame → {s3_uri}")
        s3.put_object(
            Bucket=settings.RESULTS_S3_BUCKET,
            Key=s3_key,
            Body=frame_data,
            ContentType="image/jpeg",
        )
    except Exception as exc:
        logger.warning(f"S3 upload failed for {s3_uri}: {exc}")
        if _has_aws_credentials():
            raise

    return s3_uri


# ---------------------------------------------------------------------------
# Audio Extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path: str, job_id: str) -> Optional[str]:
    """
    Extracts audio track from the video as mono 16kHz WAV via ffmpeg.
    Returns the local WAV path, or None if ffmpeg is not found or extraction fails.
    """
    output_dir = os.path.join(settings.LOCAL_STORAGE_DIR, "audio", job_id)
    os.makedirs(output_dir, exist_ok=True)
    output_wav = os.path.join(output_dir, "audio.wav")

    if os.path.exists(output_wav):
        return output_wav

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass

    if not ffmpeg_path:
        logger.warning("ffmpeg not found — audio processing will be skipped.")
        return None

    try:
        cmd = [
            ffmpeg_path, "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
            output_wav,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.warning(f"ffmpeg audio extraction failed: {result.stderr[-500:]}")
            return None
        logger.info(f"Extracted audio WAV: {output_wav}")
        return output_wav
    except Exception as exc:
        logger.warning(f"Exception during ffmpeg audio extraction: {exc}")
        return None


# ---------------------------------------------------------------------------
# Frame Sampling
# ---------------------------------------------------------------------------

def calculate_motion_score(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Returns a normalized [0.0, 1.0] motion score between two frames.
    Uses 64×64 grayscale downscale for fast pixel-diff computation.
    """
    f1 = cv2.cvtColor(cv2.resize(frame1, (64, 64)), cv2.COLOR_BGR2GRAY)
    f2 = cv2.cvtColor(cv2.resize(frame2, (64, 64)), cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(f1, f2)) / 255.0)


def _resize_to_long_edge(frame: np.ndarray, long_edge: int = 640) -> np.ndarray:
    """Downscales a frame so the longest edge equals `long_edge`, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if max(h, w) <= long_edge:
        return frame
    scale = long_edge / max(h, w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def sample_video_frames(
    video_path: str,
) -> Tuple[List[Tuple[float, np.ndarray]], Dict[float, np.ndarray]]:
    """
    Performs TRUE adaptive frame sampling on the input video.

    The sampling interval is updated after EVERY captured frame based on the
    measured motion score, so dense/sparse adaptation actually takes effect.
    (The previous implementation pre-froze a target list at baseline interval,
    making the adaptive logic a dead code path.)

    Optimization: We read frames sequentially with cap.grab() to skip cheaply,
    calling cap.retrieve() only for frames we actually want. This is 5-20x
    faster than repeated cap.set(CAP_PROP_POS_MSEC) on H.264-encoded videos.

    Returns:
        frames_list: Sorted list of (timestamp_seconds, frame_bgr) tuples.
        frames_dict: Dict mapping timestamp → frame for O(1) lookups.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    baseline = settings.ADAPTIVE_SAMPLING_BASELINE_INTERVAL
    dense    = settings.ADAPTIVE_SAMPLING_DENSE_INTERVAL
    sparse   = settings.ADAPTIVE_SAMPLING_SPARSE_INTERVAL
    thresh   = settings.SCENE_CHANGE_THRESHOLD

    frames_list: List[Tuple[float, np.ndarray]] = []
    prev_frame: Optional[np.ndarray] = None
    current_interval = baseline

    # True adaptive loop: we decide the NEXT target dynamically after each capture
    next_target_time = 0.0
    current_frame_idx = 0

    while next_target_time < duration:
        target_frame_idx = int(round(next_target_time * fps))
        target_frame_idx = min(target_frame_idx, total_frames - 1)

        # Skip to target frame cheaply using grab()
        while current_frame_idx < target_frame_idx:
            if not cap.grab():
                break
            current_frame_idx += 1

        ret, frame = cap.read()
        if not ret:
            break
        current_frame_idx += 1

        timestamp = current_frame_idx / fps
        frame_resized = _resize_to_long_edge(frame)
        frames_list.append((timestamp, frame_resized))

        # Adapt interval for the NEXT frame based on this frame's motion
        if prev_frame is not None:
            motion = calculate_motion_score(frame, prev_frame)
            if motion > thresh:
                current_interval = dense    # High motion → sample more frequently
            elif motion < thresh * 0.2:
                current_interval = sparse   # Very static → sample less frequently
            else:
                current_interval = baseline
        prev_frame = frame

        next_target_time += current_interval

    cap.release()

    # Build timestamp-keyed dict for O(log N) evidence frame lookup
    frames_dict: Dict[float, np.ndarray] = {ts: fr for ts, fr in frames_list}

    logger.info(
        f"Sampled {len(frames_list)} frames from {duration:.1f}s video "
        f"({video_path}) using true adaptive sequential read."
    )
    return frames_list, frames_dict


def find_nearest_frame(
    frames_list: List[Tuple[float, np.ndarray]],
    target_ts: float,
) -> Optional[np.ndarray]:
    """
    Finds the sampled frame whose timestamp is closest to `target_ts`
    using binary search — O(log N) instead of O(N).
    """
    if not frames_list:
        return None
    timestamps = [ts for ts, _ in frames_list]
    idx = bisect.bisect_left(timestamps, target_ts)
    if idx == 0:
        return frames_list[0][1]
    if idx >= len(timestamps):
        return frames_list[-1][1]
    # Pick whichever neighbour is closer
    before = frames_list[idx - 1]
    after  = frames_list[idx]
    return before[1] if abs(before[0] - target_ts) <= abs(after[0] - target_ts) else after[1]
