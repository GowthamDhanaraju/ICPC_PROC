import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # AWS & S3 Settings
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    SOURCE_S3_BUCKET: str = "proctoring-incoming"
    RESULTS_S3_BUCKET: str = "proctoring-results"
    
    # Database Settings
    DATABASE_URL: str = "sqlite:///./proctoring.db"
    
    # ML & Proctoring Behavior
    MOCK_ML_MODELS: bool = True  # Set to False to run actual models
    
    # Visual Thresholds
    FACE_CONFIDENCE_THRESHOLD: float = 0.5
    GADGET_CONFIDENCE_THRESHOLD: float = 0.5
    GAZE_AWAY_PITCH_THRESHOLD: float = 15.0  # degrees
    GAZE_AWAY_YAW_THRESHOLD: float = 20.0    # degrees
    
    # Preprocessing & Sampling (seconds)
    ADAPTIVE_SAMPLING_BASELINE_INTERVAL: float = 1.6
    ADAPTIVE_SAMPLING_DENSE_INTERVAL: float = 0.4
    ADAPTIVE_SAMPLING_SPARSE_INTERVAL: float = 4.0
    SCENE_CHANGE_THRESHOLD: float = 0.02
    
    # Webhook Config
    WEBHOOK_TIMEOUT: float = 10.0
    WEBHOOK_MAX_RETRIES: int = 3
    
    # Local Directories for storage mocks (when S3 is not configured or during local testing)
    LOCAL_STORAGE_DIR: str = "./storage"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure local storage mock directories exist
if not os.path.exists(settings.LOCAL_STORAGE_DIR):
    os.makedirs(settings.LOCAL_STORAGE_DIR, exist_ok=True)
    os.makedirs(os.path.join(settings.LOCAL_STORAGE_DIR, settings.SOURCE_S3_BUCKET), exist_ok=True)
    os.makedirs(os.path.join(settings.LOCAL_STORAGE_DIR, settings.RESULTS_S3_BUCKET), exist_ok=True)
