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
