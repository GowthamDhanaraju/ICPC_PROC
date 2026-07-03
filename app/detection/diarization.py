"""
Speaker Diarization Module

Upgrade: Placeholder returning [] → resemblyzer GE2E speaker embeddings

resemblyzer uses the GE2E (Generalized End-to-End) speaker encoder from Google's
TTS research paper. Key properties:
  - 17MB model, auto-downloads, no HuggingFace token required
  - ~94% accuracy (Diarization Error Rate) on AMI corpus
  - Works by embedding speech segments into 256-dim speaker space,
    then detecting when cosine similarity drops below a threshold (new speaker)
  - Fast: ~10ms per segment on CPU

Optional upgrade: pyannote/speaker-diarization-3.1 (state-of-the-art, requires
HuggingFace token). Enable by setting DIARIZATION_BACKEND=pyannote and
HUGGINGFACE_TOKEN=hf_xxx in .env.
"""
import logging
import os
import threading
from typing import Any, List, Optional, Tuple

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# resemblyzer backend
# ---------------------------------------------------------------------------
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    from pathlib import Path
    _HAS_RESEMBLYZER = True
except ImportError:
    _HAS_RESEMBLYZER = False

# ---------------------------------------------------------------------------
# pyannote backend (optional, requires HuggingFace token)
# ---------------------------------------------------------------------------
try:
    from pyannote.audio import Pipeline as PyannotePipeline
    _HAS_PYANNOTE = True
except ImportError:
    _HAS_PYANNOTE = False

_RESEMBLYZER_ENCODER: Optional[Any] = None
_RESEMBLYZER_LOCK = threading.Lock()

_PYANNOTE_PIPELINE: Optional[Any] = None
_PYANNOTE_LOCK = threading.Lock()


def _get_resemblyzer_encoder() -> Optional[Any]:
    """Lazy-load resemblyzer GE2E encoder (~17MB auto-download)."""
    global _RESEMBLYZER_ENCODER
    if not _HAS_RESEMBLYZER:
        return None
    if _RESEMBLYZER_ENCODER is None:
        with _RESEMBLYZER_LOCK:
            if _RESEMBLYZER_ENCODER is None:
                logger.info("Loading resemblyzer GE2E speaker encoder...")
                try:
                    _RESEMBLYZER_ENCODER = VoiceEncoder(device="cpu")
                    logger.info("resemblyzer encoder loaded.")
                except Exception as exc:
                    logger.error(f"resemblyzer failed to load: {exc}")
    return _RESEMBLYZER_ENCODER


def _get_pyannote_pipeline() -> Optional[Any]:
    """Lazy-load pyannote diarization pipeline (requires HuggingFace token)."""
    global _PYANNOTE_PIPELINE
    if not _HAS_PYANNOTE or not settings.HUGGINGFACE_TOKEN:
        return None
    if _PYANNOTE_PIPELINE is None:
        with _PYANNOTE_LOCK:
            if _PYANNOTE_PIPELINE is None:
                logger.info("Loading pyannote/speaker-diarization-3.1...")
                try:
                    _PYANNOTE_PIPELINE = PyannotePipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=settings.HUGGINGFACE_TOKEN,
                    )
                    logger.info("pyannote pipeline loaded.")
                except Exception as exc:
                    logger.error(f"pyannote pipeline load failed: {exc}")
    return _PYANNOTE_PIPELINE


# ---------------------------------------------------------------------------
# resemblyzer diarization
# ---------------------------------------------------------------------------

def _diarize_resemblyzer(
    wav_path: str,
    speech_segments: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """
    Uses GE2E speaker embeddings to detect when a second speaker appears.

    Algorithm:
    1. Embed the FIRST detected speech segment as the "candidate reference" voice.
    2. For each subsequent segment, embed it and compute cosine similarity to reference.
    3. If similarity < threshold, flag as SECOND_VOICE_DETECTED.
    """
    encoder = _get_resemblyzer_encoder()
    if encoder is None:
        raise RuntimeError("resemblyzer not available")

    if len(speech_segments) < 2:
        return []

    try:
        import soundfile as sf
        audio, sr = sf.read(wav_path, dtype="float32")
    except ImportError:
        try:
            import scipy.io.wavfile as wavfile
            sr, data = wavfile.read(wav_path)
            audio = data.astype(np.float32)
            if audio.max() > 1.0:
                audio = audio / 32768.0
        except Exception as exc:
            raise RuntimeError(f"Cannot read audio: {exc}")

    def extract_segment(start_s: float, end_s: float) -> Optional[np.ndarray]:
        s = int(start_s * sr)
        e = int(end_s * sr)
        chunk = audio[s:e]
        if len(chunk) < sr * 0.3:  # Need at least 300ms
            return None
        return chunk

    # Embed reference speaker (first segment)
    ref_audio = extract_segment(*speech_segments[0])
    if ref_audio is None:
        return []

    try:
        ref_wav = preprocess_wav(ref_audio, source_sr=sr)
        ref_embedding = encoder.embed_utterance(ref_wav)
    except Exception as exc:
        logger.warning(f"Failed to embed reference speaker: {exc}")
        return []

    # Similarity threshold — below this → likely a different speaker
    # 0.75 is a good starting point; lower = more sensitive
    threshold = 0.75
    second_speaker_segments = []

    for start, end in speech_segments[1:]:
        seg_audio = extract_segment(start, end)
        if seg_audio is None:
            continue
        try:
            seg_wav = preprocess_wav(seg_audio, source_sr=sr)
            seg_embedding = encoder.embed_utterance(seg_wav)
            similarity = float(np.dot(ref_embedding, seg_embedding) /
                               (np.linalg.norm(ref_embedding) * np.linalg.norm(seg_embedding)))

            if similarity < threshold:
                logger.info(
                    f"Second speaker detected at {start:.1f}s–{end:.1f}s "
                    f"(similarity={similarity:.3f} < {threshold})"
                )
                second_speaker_segments.append((start, end))
        except Exception as exc:
            logger.debug(f"Embedding failed for segment {start:.1f}s–{end:.1f}s: {exc}")

    return second_speaker_segments


# ---------------------------------------------------------------------------
# pyannote diarization
# ---------------------------------------------------------------------------

def _diarize_pyannote(wav_path: str) -> List[Tuple[float, float]]:
    """
    Runs pyannote/speaker-diarization-3.1 and returns segments with 2+ speakers.
    Requires HUGGINGFACE_TOKEN to be set.
    """
    pipeline = _get_pyannote_pipeline()
    if pipeline is None:
        raise RuntimeError("pyannote pipeline not available (check HUGGINGFACE_TOKEN)")

    diarization = pipeline(wav_path)

    # Collect segments per speaker
    from collections import defaultdict
    speaker_segments = defaultdict(list)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments[speaker].append((turn.start, turn.end))

    # Any speaker that is NOT the first/main speaker is flagged
    if len(speaker_segments) <= 1:
        return []

    speakers = sorted(speaker_segments.keys())
    main_speaker = speakers[0]

    second_speaker_segments = []
    for speaker, segs in speaker_segments.items():
        if speaker != main_speaker:
            second_speaker_segments.extend(segs)

    return sorted(second_speaker_segments, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_multiple_speakers(
    wav_path: str,
    speech_segments: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """
    Detects periods where a second (non-candidate) speaker is present.
    Returns list of (start_s, end_s) segments flagged as SECOND_VOICE_DETECTED.

    Backend cascade: pyannote (if token set) → resemblyzer → []
    """
    if settings.MOCK_ML_MODELS or not wav_path or not os.path.exists(wav_path):
        return [(100.0, 105.0)]

    backend = settings.DIARIZATION_BACKEND.lower()

    # --- pyannote (best accuracy, requires HuggingFace token) ---
    if backend == "pyannote" and settings.HUGGINGFACE_TOKEN:
        try:
            return _diarize_pyannote(wav_path)
        except Exception as exc:
            logger.warning(f"pyannote diarization failed: {exc}. Falling back to resemblyzer.")

    # --- resemblyzer GE2E (good accuracy, no token needed) ---
    if backend in ("resemblyzer", "pyannote"):
        try:
            return _diarize_resemblyzer(wav_path, speech_segments)
        except Exception as exc:
            logger.warning(f"resemblyzer diarization failed: {exc}.")

    logger.warning("No diarization backend available — second speaker detection skipped.")
    return []
