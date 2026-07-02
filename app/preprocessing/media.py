import os
import urllib.parse
import shutil
import subprocess
import cv2
import numpy as np
import boto3
from botocore.exceptions import NoCredentialsError
from typing import List, Tuple, Optional
from app.config import settings

def get_s3_client():
    """
    Initializes the boto3 S3 client with configured AWS credentials.
    Falls back to environment variables/IAM roles if credentials are not explicitly set in config.
    """
    session_kwargs = {}
    if settings.AWS_ACCESS_KEY_ID:
        session_kwargs['aws_access_key_id'] = settings.AWS_ACCESS_KEY_ID
    if settings.AWS_SECRET_ACCESS_KEY:
        session_kwargs['aws_secret_access_key'] = settings.AWS_SECRET_ACCESS_KEY
    if settings.AWS_REGION:
        session_kwargs['region_name'] = settings.AWS_REGION
    return boto3.client('s3', **session_kwargs)

def get_local_path(s3_uri: str) -> str:
    """
    Resolves an S3 URI to a local path.
    1. If it's already a valid local path, returns it.
    2. Attempts to download from S3 via boto3.
    3. If credentials exist, raises errors on failure.
    4. Falls back to mock files ONLY if running in local offline mode without credentials.
    """
    if os.path.exists(s3_uri):
        return s3_uri

    parsed = urllib.parse.urlparse(s3_uri)
    if parsed.scheme != "s3":
        return s3_uri

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    # Ensure parent directory exists for download
    download_path = os.path.join(settings.LOCAL_STORAGE_DIR, "downloads", bucket, key)
    os.makedirs(os.path.dirname(download_path), exist_ok=True)

    # Check if download cache matches
    if os.path.exists(download_path):
        # We can reuse the cache, or we can check actual S3. Let's return the cache for speed,
        # but let the worker know
        return download_path

    # Try downloading from actual S3
    try:
        s3 = get_s3_client()
        print(f"Downloading s3://{bucket}/{key} to local path {download_path}...")
        s3.download_file(bucket, key, download_path)
        return download_path
    except Exception as e:
        print(f"S3 download failed for {s3_uri}: {e}")
        # Detect if any AWS configurations or credentials exist
        has_creds = (
            settings.AWS_ACCESS_KEY_ID is not None or 
            os.getenv("AWS_ACCESS_KEY_ID") is not None or 
            os.path.exists(os.path.expanduser("~/.aws/credentials")) or
            os.path.exists(os.path.expanduser("~/.aws/config"))
        )
        if has_creds:
            raise e
            
        # Fall back to local mock video if offline and no credentials exist
        # Check local storage simulation folder first
        local_mock_path = os.path.join(settings.LOCAL_STORAGE_DIR, bucket, key)
        if os.path.exists(local_mock_path):
            return local_mock_path
            
        fallback = os.path.join(settings.LOCAL_STORAGE_DIR, "mock_video.mp4")
        if os.path.exists(fallback):
            print(f"AWS credentials not detected. Falling back to local mock video: {fallback}")
            return fallback
        raise FileNotFoundError(f"S3 file {s3_uri} could not be retrieved and no local fallback video exists. Original error: {e}")

def upload_evidence_frame(job_id: str, timestamp: float, frame_data: bytes) -> str:
    """
    Uploads an evidence frame image to the results S3 bucket.
    Raises exception on failure if AWS credentials are set.
    """
    filename = f"evidence_{job_id}_{int(timestamp * 1000)}.jpg"
    s3_key = f"results/{job_id}/{filename}"
    
    # Save locally first
    local_path = os.path.join(settings.LOCAL_STORAGE_DIR, settings.RESULTS_S3_BUCKET, job_id, filename)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(frame_data)

    # Try uploading to actual S3
    try:
        s3 = get_s3_client()
        print(f"Uploading evidence frame to s3://{settings.RESULTS_S3_BUCKET}/{s3_key}...")
        s3.put_object(
            Bucket=settings.RESULTS_S3_BUCKET,
            Key=s3_key,
            Body=frame_data,
            ContentType="image/jpeg"
        )
        return f"s3://{settings.RESULTS_S3_BUCKET}/{s3_key}"
    except Exception as e:
        print(f"S3 upload failed for s3://{settings.RESULTS_S3_BUCKET}/{s3_key}: {e}")
        # Detect if any AWS configurations or credentials exist
        has_creds = (
            settings.AWS_ACCESS_KEY_ID is not None or 
            os.getenv("AWS_ACCESS_KEY_ID") is not None or 
            os.path.exists(os.path.expanduser("~/.aws/credentials")) or
            os.path.exists(os.path.expanduser("~/.aws/config"))
        )
        if has_creds:
            raise e
            
        # Return mock S3 URI for local offline testing if no credentials exist
        return f"s3://{settings.RESULTS_S3_BUCKET}/{s3_key}"

def extract_audio(video_path: str, job_id: str) -> Optional[str]:
    """
    Extracts the audio track from the video as mono, 16kHz WAV format.
    Returns the local path to the extracted WAV file, or None if extraction fails.
    """
    output_dir = os.path.join(settings.LOCAL_STORAGE_DIR, "audio", job_id)
    os.makedirs(output_dir, exist_ok=True)
    output_wav = os.path.join(output_dir, "audio.wav")

    if os.path.exists(output_wav):
        return output_wav

    # Locate ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass

    if not ffmpeg_path:
        print("Warning: ffmpeg not found. Audio processing will be skipped.")
        return None

    try:
        cmd = [
            ffmpeg_path,
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            output_wav
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Warning: ffmpeg audio extraction failed: {result.stderr}")
            return None
        return output_wav
    except Exception as e:
        print(f"Warning: Exception while running ffmpeg: {e}")
        return None

def calculate_motion_score(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Computes a normalized motion score between two frames based on pixel-wise difference.
    Resizes both frames to 64x64 grayscale for fast computation.
    """
    f1_gray = cv2.cvtColor(cv2.resize(frame1, (64, 64)), cv2.COLOR_BGR2GRAY)
    f2_gray = cv2.cvtColor(cv2.resize(frame2, (64, 64)), cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(f1_gray, f2_gray)
    return float(np.mean(diff) / 255.0)

def sample_video_frames(video_path: str) -> List[Tuple[float, np.ndarray]]:
    """
    Performs adaptive frame sampling on the input video:
    - Analyzes motion/scene differences between subsequent frames.
    - Densifies sampling during high motion.
    - Sparsifies sampling during low motion.
    - Downscales frames to 640px on the long edge.
    Returns a list of (timestamp_seconds, frame_bgr_array).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Fallback
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    sampled_frames = []
    
    current_time = 0.0
    prev_frame = None
    
    # Adaptive intervals
    baseline = settings.ADAPTIVE_SAMPLING_BASELINE_INTERVAL
    dense = settings.ADAPTIVE_SAMPLING_DENSE_INTERVAL
    sparse = settings.ADAPTIVE_SAMPLING_SPARSE_INTERVAL
    scene_threshold = settings.SCENE_CHANGE_THRESHOLD

    current_interval = baseline

    while current_time < duration:
        # Seek to the target timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000.0)
        ret, frame = cap.read()
        if not ret:
            break

        # Downscale frame
        h, w = frame.shape[:2]
        long_edge = 640
        if max(h, w) > long_edge:
            if w > h:
                new_w = long_edge
                new_h = int(h * (long_edge / w))
            else:
                new_h = long_edge
                new_w = int(w * (long_edge / h))
            frame_resized = cv2.resize(frame, (new_w, new_h))
        else:
            frame_resized = frame.copy()

        sampled_frames.append((current_time, frame_resized))

        # Adjust the sampling rate adaptively
        if prev_frame is not None:
            motion = calculate_motion_score(frame, prev_frame)
            if motion > scene_threshold:
                # High motion -> sample more frequently (dense)
                current_interval = dense
            elif motion < scene_threshold * 0.2:
                # Very low motion -> sample less frequently (sparse)
                current_interval = sparse
            else:
                # Normal motion -> baseline
                current_interval = baseline
        else:
            current_interval = baseline

        prev_frame = frame
        current_time += current_interval

    cap.release()
    print(f"Sampled {len(sampled_frames)} frames from video of length {duration:.2f}s (Adaptive)")
    return sampled_frames
