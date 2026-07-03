"""
Centralized logging configuration for the Batch Video Proctoring Pipeline.
Supports plain text (development) and JSON-structured (production) output,
compatible with Datadog, CloudWatch, and any log aggregation pipeline.
"""
import logging
import logging.config
import json
import datetime
import sys
from app.config import settings


class JsonFormatter(logging.Formatter):
    """
    Emits log records as single-line JSON objects.
    This format is natively parseable by CloudWatch Insights, Datadog, etc.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def configure_logging():
    """
    Applies the project-wide logging configuration.
    Call once at application startup (in main.py lifespan).
    """
    is_production = settings.ENV == "production"

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JsonFormatter,
            },
            "plain": {
                "format": "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "json" if is_production else "plain",
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["console"],
        },
        "loggers": {
            # Silence noisy third-party libraries
            "uvicorn.access": {"level": "WARNING", "propagate": True},
            "boto3": {"level": "WARNING", "propagate": True},
            "botocore": {"level": "WARNING", "propagate": True},
            "urllib3": {"level": "WARNING", "propagate": True},
            "ultralytics": {"level": "WARNING", "propagate": True},
            "mediapipe": {"level": "WARNING", "propagate": True},
        },
    })

    logging.getLogger(__name__).info(
        f"Logging configured. Environment: {settings.ENV}. "
        f"Format: {'JSON' if is_production else 'plaintext'}"
    )
