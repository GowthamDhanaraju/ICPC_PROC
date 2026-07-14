"""
Voice Activity Detection (VAD) Module

Backend: Silero VAD (PyTorch).
  - 97.4% F1 on RealWorld VAD — best available lightweight model.
  - Handles background noise, music, coughing, keyboard clicks without false triggers.
  - <1 ms per 30 ms audio chunk (negligible overhead).
  - Model auto-downloads (~5 MB) via torch.hub on first use.

Speech probability threshold is set to 0.65 (raised from the default 0.50)
to suppress borderline detections that could generate false-positive voice flags.
"""
import logging
import os
import threading
from typing import Any, List, Optional, Tuple

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Torch dependency check
# ---------------------------------------------------------------------------
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    logger.error(
        "PyTorch is not installed. Silero VAD will be unavailable. "
        "Install it with: pip install torch"
    )

# ---------------------------------------------------------------------------
# Silero VAD — PyTorch (sole backend)
# ---------------------------------------------------------------------------
_SILERO_MODEL: Optional[Any] = None
_SILERO_UTILS: Optional[Any] = None
_SILERO_LOCK = threading.Lock()

# Speech probability threshold — higher = fewer, more confident detections.
_VAD_THRESHOLD = 0.65   # raised from 0.50 for stricter false-positive suppression
_SILERO_SAMPLE_RATE = 16000


def _get_silero_model() -> Tuple[Optional[Any], Optional[Any]]:
    """Load Silero VAD via torch.hub (downloads ~5 MB on first use)."""
    global _SILERO_MODEL, _SILERO_UTILS
    if not _HAS_TORCH:
        return None, None
    if _SILERO_MODEL is None:
        with _SILERO_LOCK:
            if _SILERO_MODEL is None:
                try:
                    logger.info("Loading Silero VAD (PyTorch)...")
                    model, utils = torch.hub.load(
                        repo_or_dir="snakers4/silero-vad",
                        model="silero_vad",
                        force_reload=False,
                        trust_repo=True,
                    )
                    _SILERO_MODEL = model
                    _SILERO_UTILS = utils
                    logger.info("Silero VAD loaded.")
                except Exception as exc:
                    logger.error(f"Silero VAD load failed: {exc}")
    return _SILERO_MODEL, _SILERO_UTILS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_voice_activity(wav_path: str) -> List[Tuple[float, float]]:
    """
    Analyses a 16 kHz mono WAV file and returns (start_s, end_s) speech segments.

    Uses Silero VAD with a strict probability threshold of {threshold} to
    minimise false-positive speech detections.
    """.format(threshold=_VAD_THRESHOLD)
    if settings.MOCK_ML_MODELS or not wav_path or not os.path.exists(wav_path):
        return [(10.0, 12.0), (60.0, 65.0), (100.0, 105.0), (140.0, 142.0)]

    try:
        import scipy.io.wavfile as wavfile
        sample_rate, data = wavfile.read(wav_path)
    except Exception as exc:
        logger.error(f"Failed to read WAV file {wav_path}: {exc}")
        return []

    model, utils = _get_silero_model()
    if model is None or utils is None:
        logger.error("Silero VAD unavailable — voice activity detection skipped.")
        return []

    try:
        get_speech_ts = utils[0]  # first util is get_speech_timestamps
        audio_tensor = torch.from_numpy(data).float()
        if len(audio_tensor.shape) == 2:
            audio_tensor = audio_tensor.mean(0)

        speech_timestamps = get_speech_ts(
            audio_tensor,
            model,
            sampling_rate=sample_rate,
            threshold=_VAD_THRESHOLD,
            min_speech_duration_ms=400,   # ignore very short utterances (raised from 300 ms)
            min_silence_duration_ms=150,  # require clear silences between segments
        )
        return [(ts["start"] / sample_rate, ts["end"] / sample_rate) for ts in speech_timestamps]
    except Exception as exc:
        logger.error(f"Silero VAD inference failed: {exc}")
        return []
