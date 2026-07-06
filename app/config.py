"""
Application configuration loaded from environment variables / .env file.
All settings can be overridden at runtime without code changes.
"""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # --- Application Environment ---
    ENV: str = "development"  # "development" | "production"

    # --- AWS S3 ---
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    SOURCE_S3_BUCKET: str = "proctoring-incoming"
    RESULTS_S3_BUCKET: str = "proctoring-results"

    # --- MinIO (S3-compatible object storage) ---
    # Set MINIO_ENDPOINT to point at your MinIO server (e.g. "localhost:9000").
    # When set, all uploads and downloads are routed through MinIO instead of AWS S3.
    MINIO_ENDPOINT: Optional[str] = None          # host:port, no scheme
    MINIO_ACCESS_KEY: Optional[str] = None        # MinIO root user / access key
    MINIO_SECRET_KEY: Optional[str] = None        # MinIO root password / secret key
    MINIO_USE_SSL: bool = False                    # Set True when MinIO is behind HTTPS

    # --- Database ---
    DATABASE_URL: str = "sqlite:///./proctoring.db"

    # --- Celery / Redis Worker Queue ---
    # Leave empty to use the in-process thread queue (development only)
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # --- ML Model Toggle ---
    MOCK_ML_MODELS: bool = True  # Set to False on production server

    # --- Model Backend Selection ---
    # Face detection backend: "insightface" | "yolo" | "haar"
    FACE_BACKEND: str = "insightface"
    # Identity verification backend: "arcface" | "lbph"
    IDENTITY_BACKEND: str = "arcface"
    # Gaze/head pose backend: "6drepnet" | "mediapipe"
    GAZE_BACKEND: str = "6drepnet"
    # VAD backend: "silero" | "rms"
    VAD_BACKEND: str = "silero"
    # Diarization backend: "resemblyzer" | "pyannote"
    DIARIZATION_BACKEND: str = "resemblyzer"

    # --- HuggingFace (optional, for pyannote diarization) ---
    HUGGINGFACE_TOKEN: Optional[str] = None

    # --- InsightFace GPU context (-1 = CPU) ---
    INSIGHTFACE_CTX_ID: int = -1

    # --- Visual Detection Thresholds ---
    FACE_CONFIDENCE_THRESHOLD: float = 0.8
    GADGET_CONFIDENCE_THRESHOLD: float = 0.8
    GAZE_AWAY_PITCH_THRESHOLD: float = 15.0   # degrees
    GAZE_AWAY_YAW_THRESHOLD: float = 20.0      # degrees

    # --- Adaptive Frame Sampling (seconds) ---
    ADAPTIVE_SAMPLING_BASELINE_INTERVAL: float = 1.6
    ADAPTIVE_SAMPLING_DENSE_INTERVAL: float = 0.4
    ADAPTIVE_SAMPLING_SPARSE_INTERVAL: float = 4.0
    SCENE_CHANGE_THRESHOLD: float = 0.02

    # --- Webhook ---
    WEBHOOK_TIMEOUT: float = 10.0
    WEBHOOK_MAX_RETRIES: int = 3

    # --- Worker ---
    WORKER_CONCURRENCY: int = 2  # Number of in-process thread workers

    # --- Testing ---
    # When True, /test/* routes are mounted (never enable in production)
    TESTING_MODE: bool = False

    # --- Debug Features ---
    # When True, generates an MP4 with bounding boxes/gaze lines and uploads it to MinIO
    RENDER_OVERLAY_VIDEO: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
