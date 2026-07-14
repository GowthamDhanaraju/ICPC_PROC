"""
Video preprocessing and MinIO/S3 storage utilities.

Storage model:
- ALL persistent artifacts (evidence frames, audio downloads) go to MinIO/S3.
- No files are written to local disk permanently.
- Temp files are created in a per-job temp directory and cleaned up by the
  worker after each job completes (success or failure).

MinIO is boto3-compatible: set MINIO_ENDPOINT (host:port) in .env and boto3
will use it instead of AWS S3.  AWS credentials (MINIO_ACCESS_KEY /
MINIO_SECRET_KEY) are passed through the standard boto3 mechanism.
"""
import bisect
import logging
import os
import shutil
import subprocess
import tempfile
import urllib.parse
from typing import Dict, List, Optional, Tuple

import boto3
import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# S3 / MinIO Client
# ---------------------------------------------------------------------------

def get_s3_client():
    """
    Returns a boto3 S3 client pointing at MinIO (if MINIO_ENDPOINT is set)
    or AWS S3 (otherwise).  Credentials fall back to IAM role / ~/.aws config
    when neither MINIO nor explicit AWS keys are configured.
    """
    kwargs: Dict = {}

    if settings.MINIO_ENDPOINT:
        # MinIO: always use path-style addressing (virtual-hosted style is default on AWS)
        scheme = "https" if settings.MINIO_USE_SSL else "http"
        kwargs["endpoint_url"] = f"{scheme}://{settings.MINIO_ENDPOINT}"
        kwargs["config"] = boto3.session.Config(signature_version="s3v4")
        if settings.MINIO_ACCESS_KEY:
            kwargs["aws_access_key_id"] = settings.MINIO_ACCESS_KEY
        if settings.MINIO_SECRET_KEY:
            kwargs["aws_secret_access_key"] = settings.MINIO_SECRET_KEY
        # MinIO doesn't use AWS regions but boto3 requires a value
        kwargs["region_name"] = "us-east-1"
    else:
        # AWS S3
        kwargs["region_name"] = settings.AWS_REGION
        if settings.AWS_ACCESS_KEY_ID:
            kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
        if settings.AWS_SECRET_ACCESS_KEY:
            kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY

    return boto3.client("s3", **kwargs)


def _is_object_storage_configured() -> bool:
    """
    Returns True when object storage (MinIO or AWS S3) is reachable.
    Used to decide whether to attempt uploads.
    """
    return (
        settings.MINIO_ENDPOINT is not None
        or settings.AWS_ACCESS_KEY_ID is not None
        or os.getenv("AWS_ACCESS_KEY_ID") is not None
        or os.path.exists(os.path.expanduser("~/.aws/credentials"))
        or os.path.exists(os.path.expanduser("~/.aws/config"))
    )


# ---------------------------------------------------------------------------
# Object Storage Helpers
# ---------------------------------------------------------------------------

def get_local_path(s3_uri: str, temp_dir: Optional[str] = None) -> str:
    """
    Resolves an S3/MinIO URI to a local filesystem path for processing.

    - If the URI is already a local path that exists, returns it as-is (no copy).
    - Otherwise downloads the object to a temp file inside `temp_dir`
      (or a system temp file if temp_dir is None).

    Within a job's temp_dir the same filename is reused on repeated calls
    (natural download cache), so the enrollment photo is only fetched once
    per job regardless of how many frames call verify_identity().

    **Caller must NOT delete files returned for pre-existing local paths.**
    All files placed inside temp_dir are cleaned up by the worker's finally block.
    """
    # Already a local path — return directly without copying
    if os.path.exists(s3_uri):
        return s3_uri

    parsed = urllib.parse.urlparse(s3_uri)
    if parsed.scheme != "s3":
        # Treat as a literal path (might not exist — let the caller handle the error)
        return s3_uri

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    # Determine where to download
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
        filename = os.path.basename(key) or "download"
        download_path = os.path.join(temp_dir, filename)
        # Reuse within the same job temp dir (acts as a per-job download cache)
        if os.path.exists(download_path):
            logger.debug(f"Reusing cached download: {download_path}")
            return download_path
    else:
        suffix = os.path.splitext(key)[1] or ""
        fd, download_path = tempfile.mkstemp(suffix=suffix, prefix="proctoring_dl_")
        os.close(fd)

    try:
        s3 = get_s3_client()
        logger.info(f"Downloading s3://{bucket}/{key} → {download_path}")
        s3.download_file(bucket, key, download_path)
        return download_path
    except Exception as exc:
        # Clean up the empty temp file we created
        if os.path.exists(download_path) and not (temp_dir and os.path.dirname(download_path) == temp_dir):
            try:
                os.unlink(download_path)
            except OSError:
                pass
        raise FileNotFoundError(
            f"Could not retrieve {s3_uri} from object storage: {exc}"
        ) from exc


def upload_evidence_frame(job_id: str, timestamp: float, frame_data: bytes) -> Optional[str]:
    """
    Uploads an evidence frame JPEG directly to MinIO/S3.
    Returns the object URI, or None if no object storage is configured.
    No local file is written.
    """
    if not _is_object_storage_configured():
        logger.warning(
            "No object storage configured (set MINIO_ENDPOINT or AWS credentials). "
            "Evidence frame not stored."
        )
        return None

    filename = f"evidence_{job_id}_{int(timestamp * 1000)}.jpg"
    s3_key = f"results/{job_id}/{filename}"
    s3_uri = f"s3://{settings.RESULTS_S3_BUCKET}/{s3_key}"

    try:
        s3 = get_s3_client()
        s3.put_object(
            Bucket=settings.RESULTS_S3_BUCKET,
            Key=s3_key,
            Body=frame_data,
            ContentType="image/jpeg",
        )
        logger.info(f"Evidence frame uploaded → {s3_uri}")
        return s3_uri
    except Exception as exc:
        logger.warning(f"Evidence frame upload failed for job {job_id} at {timestamp:.1f}s: {exc}")
        return None


def upload_report(job_id: str, filename: str, report_json_str: str) -> Optional[str]:
    """
    Uploads the JSON report to MinIO/S3.
    Returns the object URI, or None if no object storage is configured.
    """
    if not _is_object_storage_configured():
        return None

    s3_key = f"results/{job_id}/{filename}"
    s3_uri = f"s3://{settings.RESULTS_S3_BUCKET}/{s3_key}"

    try:
        s3 = get_s3_client()
        s3.put_object(
            Bucket=settings.RESULTS_S3_BUCKET,
            Key=s3_key,
            Body=report_json_str.encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(f"JSON report uploaded → {s3_uri}")
        return s3_uri
    except Exception as exc:
        logger.warning(f"JSON report upload failed for job {job_id}: {exc}")
        return None






# ---------------------------------------------------------------------------
# Audio Extraction
# ---------------------------------------------------------------------------

def extract_audio(
    video_path: str,
    job_id: str,
    temp_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Extracts audio from the video as a mono 16 kHz WAV using ffmpeg.

    The WAV is written to a temp file inside `temp_dir` (or a system temp file
    if temp_dir is None).  The worker's finally block cleans up temp_dir.

    Returns the temp WAV path, or None if ffmpeg is unavailable.
    """
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
        output_wav = os.path.join(temp_dir, "audio.wav")
        # Reuse if already extracted in this job's temp dir
        if os.path.exists(output_wav):
            return output_wav
    else:
        fd, output_wav = tempfile.mkstemp(suffix=".wav", prefix=f"proctoring_{job_id[:8]}_")
        os.close(fd)

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

def slice_and_upload_audio(
    audio_path: str,
    start_ts: float,
    end_ts: float,
    job_id: str,
    filename: str,
    temp_dir: Optional[str] = None
) -> Optional[str]:
    """
    Slices a portion of an audio WAV file using ffmpeg, then uploads it to MinIO.
    Returns the uploaded S3 URI, or None on failure.
    """
    if not _is_object_storage_configured():
        return None

    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
        slice_path = os.path.join(temp_dir, f"{filename}")
    else:
        fd, slice_path = tempfile.mkstemp(suffix=".wav", prefix=f"proctoring_slice_")
        os.close(fd)

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass

    if not ffmpeg_path:
        logger.warning("ffmpeg not found — audio slicing skipped.")
        return None

    try:
        # Slice audio: -ss start_time -to end_time
        cmd = [
            ffmpeg_path, "-y", 
            "-i", audio_path,
            "-ss", str(start_ts),
            "-to", str(end_ts),
            "-acodec", "copy",
            slice_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.warning(f"ffmpeg audio slicing failed: {result.stderr[-500:]}")
            return None
            
        s3_key = f"results/{job_id}/{filename}"
        s3_uri = f"s3://{settings.RESULTS_S3_BUCKET}/{s3_key}"
        
        s3 = get_s3_client()
        s3.upload_file(
            slice_path,
            settings.RESULTS_S3_BUCKET,
            s3_key,
            ExtraArgs={"ContentType": "audio/wav"}
        )
        logger.info(f"Sliced audio segment uploaded → {s3_uri}")
        return s3_uri
    except Exception as exc:
        logger.warning(f"Audio slicing/upload failed: {exc}")
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

    Uses cap.grab() for sequential seeks — 5-20x faster than cap.set() on H.264.

    Returns:
        frames_list: Sorted list of (timestamp_seconds, frame_bgr) tuples.
        frames_dict: Dict mapping timestamp → frame for O(log N) evidence lookups.
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

    next_target_time = 0.0
    current_frame_idx = 0

    while next_target_time < duration:
        target_frame_idx = min(int(round(next_target_time * fps)), total_frames - 1)

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

        if prev_frame is not None:
            motion = calculate_motion_score(frame, prev_frame)
            if motion > thresh:
                current_interval = dense
            elif motion < thresh * 0.2:
                current_interval = sparse
            else:
                current_interval = baseline
        prev_frame = frame

        next_target_time += current_interval

    cap.release()

    frames_dict: Dict[float, np.ndarray] = {ts: fr for ts, fr in frames_list}

    logger.info(
        f"Sampled {len(frames_list)} frames from {duration:.1f}s video "
        f"({video_path}) using adaptive sequential read."
    )
    return frames_list, frames_dict


def find_nearest_frame(
    frames_list: List[Tuple[float, np.ndarray]],
    target_ts: float,
) -> Optional[np.ndarray]:
    """
    Finds the sampled frame whose timestamp is closest to `target_ts`
    using binary search — O(log N).
    """
    if not frames_list:
        return None
    timestamps = [ts for ts, _ in frames_list]
    idx = bisect.bisect_left(timestamps, target_ts)
    if idx == 0:
        return frames_list[0][1]
    if idx >= len(timestamps):
        return frames_list[-1][1]
    before = frames_list[idx - 1]
    after  = frames_list[idx]
    return before[1] if abs(before[0] - target_ts) <= abs(after[0] - target_ts) else after[1]
