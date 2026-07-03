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

    # --- AWS & S3 ---
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    SOURCE_S3_BUCKET: str = "proctoring-incoming"
    RESULTS_S3_BUCKET: str = "proctoring-results"

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
    # Whisper model size: "tiny.en" | "base.en" | "small.en" | "medium.en"
    WHISPER_MODEL_SIZE: str = "tiny.en"

    # --- HuggingFace (optional, for pyannote diarization) ---
    HUGGINGFACE_TOKEN: Optional[str] = None

    # --- InsightFace GPU context (-1 = CPU) ---
    INSIGHTFACE_CTX_ID: int = -1

    # --- Visual Detection Thresholds ---
    FACE_CONFIDENCE_THRESHOLD: float = 0.5
    GADGET_CONFIDENCE_THRESHOLD: float = 0.5
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

    # --- Local Storage (S3 fallback for local dev) ---
    LOCAL_STORAGE_DIR: str = "./storage"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def ensure_local_dirs(self):
        """
        Creates local mock storage directories.
        Must be called explicitly at startup, NOT at import time,
        to avoid side-effects in read-only container environments and tests.
        """
        os.makedirs(self.LOCAL_STORAGE_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.LOCAL_STORAGE_DIR, self.SOURCE_S3_BUCKET), exist_ok=True)
        os.makedirs(os.path.join(self.LOCAL_STORAGE_DIR, self.RESULTS_S3_BUCKET), exist_ok=True)


settings = Settings()
