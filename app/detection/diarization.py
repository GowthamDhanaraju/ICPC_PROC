"""
Speaker Diarization / Voice Counting Module

Backend: Resemblyzer GE2E speaker encoder.
  - 256-dim GE2E embeddings, ~17 MB, no token required, CPU-capable.
  - Agglomerative clustering (single-linkage, cosine distance) to group
    embeddings into distinct speaker clusters.

Algorithm:
  1. Run all VAD speech segments through the GE2E encoder → 256-dim embedding/segment.
  2. Build a cosine-distance matrix across all embeddings.
  3. Apply agglomerative clustering (distance_threshold=0.20, i.e. similarity ≥ 0.80)
     to group embeddings into speaker clusters.  Each cluster = one distinct voice.
  4. The primary speaker is the cluster with the most total speaking time.
     All other clusters are returned as "flagged" (non-primary voice) segments.
  5. Confidence = mean intra-cluster cosine similarity.

Clustering threshold tightened to 0.20 (was 0.25) so two voice segments must be
≥80% similar to be considered the same speaker, reducing false "same-speaker" merges.
"""
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resemblyzer — sole backend
# ---------------------------------------------------------------------------
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    _HAS_RESEMBLYZER = True
except ImportError:
    _HAS_RESEMBLYZER = False
    logger.error(
        "Resemblyzer is not installed. Speaker diarization will be unavailable. "
        "Install it with: pip install resemblyzer"
    )

_RESEMBLYZER_ENCODER: Optional[Any] = None
_RESEMBLYZER_LOCK = threading.Lock()


def _get_resemblyzer_encoder() -> Optional[Any]:
    """Lazy-load resemblyzer GE2E encoder (~17 MB, auto-downloads on first use)."""
    global _RESEMBLYZER_ENCODER
    if not _HAS_RESEMBLYZER:
        return None
    if _RESEMBLYZER_ENCODER is None:
        with _RESEMBLYZER_LOCK:
            if _RESEMBLYZER_ENCODER is None:
                logger.info("Loading resemblyzer GE2E speaker encoder...")
                try:
                    _RESEMBLYZER_ENCODER = VoiceEncoder(device="cpu")
                    logger.info("Resemblyzer encoder loaded.")
                except Exception as exc:
                    logger.error(f"Resemblyzer failed to load: {exc}")
    return _RESEMBLYZER_ENCODER


# ---------------------------------------------------------------------------
# Agglomerative clustering helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def _agglomerative_cluster(
    embeddings: np.ndarray,
    distance_threshold: float = 0.20,
) -> List[int]:
    """
    Single-linkage agglomerative clustering on cosine distance.

    distance_threshold=0.20 ↔ similarity ≥ 0.80 required to merge into same speaker.
    Tightened from 0.25 to reduce false "same-speaker" merges.

    Returns a list of cluster labels (0-indexed) of length len(embeddings).
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Build symmetric cosine-distance matrix
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = 1.0 - _cosine_similarity(embeddings[i], embeddings[j])
            dist[i, j] = d
            dist[j, i] = d

    labels = list(range(n))

    def _min_dist_between_clusters(c1: int, c2: int) -> float:
        members_c1 = [k for k, lbl in enumerate(labels) if lbl == c1]
        members_c2 = [k for k, lbl in enumerate(labels) if lbl == c2]
        return float(min(dist[i, j] for i in members_c1 for j in members_c2))

    while True:
        cluster_ids = list(set(labels))
        if len(cluster_ids) <= 1:
            break

        best_d = float("inf")
        best_pair = (cluster_ids[0], cluster_ids[1])

        for ci_idx in range(len(cluster_ids)):
            for cj_idx in range(ci_idx + 1, len(cluster_ids)):
                ci, cj = cluster_ids[ci_idx], cluster_ids[cj_idx]
                d = _min_dist_between_clusters(ci, cj)
                if d < best_d:
                    best_d = d
                    best_pair = (ci, cj)

        if best_d > distance_threshold:
            break

        ci, cj = best_pair
        labels = [ci if lbl == cj else lbl for lbl in labels]

    unique = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique)}
    return [remap[lbl] for lbl in labels]


# ---------------------------------------------------------------------------
# Core resemblyzer voice counting
# ---------------------------------------------------------------------------

def _count_voices_resemblyzer(
    wav_path: str,
    speech_segments: List[Tuple[float, float]],
) -> Dict[str, Any]:
    """
    Embeds all speech segments and clusters them into distinct speaker groups.

    Returns:
        num_speakers     — count of distinct voices (int)
        speaker_segments — dict mapping "speaker_N" → list of (start, end) tuples
        flagged_segments — segments NOT belonging to the primary speaker
        confidence       — mean intra-cluster cosine similarity (0–1)
    """
    encoder = _get_resemblyzer_encoder()
    if encoder is None:
        raise RuntimeError("Resemblyzer not available")

    if len(speech_segments) < 2:
        return {
            "num_speakers": 1,
            "speaker_segments": {"speaker_0": list(speech_segments)},
            "flagged_segments": [],
            "confidence": 1.0,
        }

    # Load audio
    try:
        import soundfile as sf
        audio, sr = sf.read(wav_path, dtype="float32")
    except Exception:
        try:
            import scipy.io.wavfile as wavfile
            sr, data = wavfile.read(wav_path)
            audio = data.astype(np.float32)
            if audio.max() > 1.0:
                audio /= 32768.0
        except Exception as exc:
            raise RuntimeError(f"Cannot read audio file: {exc}")

    def _extract_segment(start_s: float, end_s: float) -> Optional[np.ndarray]:
        s = int(start_s * sr)
        e = int(end_s * sr)
        chunk = audio[s:e]
        # Need ≥300 ms to get a reliable embedding
        return chunk if len(chunk) >= int(sr * 0.3) else None

    # Embed each segment
    valid_segments: List[Tuple[float, float]] = []
    embeddings: List[np.ndarray] = []

    for start, end in speech_segments:
        chunk = _extract_segment(start, end)
        if chunk is None:
            continue
        try:
            wav = preprocess_wav(chunk, source_sr=sr)
            emb = encoder.embed_utterance(wav)
            valid_segments.append((start, end))
            embeddings.append(emb)
        except Exception as exc:
            logger.debug(f"Embedding failed for {start:.1f}s–{end:.1f}s: {exc}")

    if not embeddings:
        return {
            "num_speakers": 1,
            "speaker_segments": {"speaker_0": list(speech_segments)},
            "flagged_segments": [],
            "confidence": 1.0,
        }

    emb_matrix = np.stack(embeddings, axis=0)  # (N, 256)

    # Cluster with tightened threshold (0.20 → similarity ≥ 0.80 = same speaker)
    labels = _agglomerative_cluster(emb_matrix, distance_threshold=0.20)
    num_speakers = max(labels) + 1

    logger.info(
        f"Voice counting: {len(valid_segments)} segments → {num_speakers} distinct speaker(s)"
    )

    # Build per-speaker segment lists
    speaker_segments: Dict[str, List[Tuple[float, float]]] = {}
    speaker_duration: Dict[int, float] = {}

    for seg, label in zip(valid_segments, labels):
        key = f"speaker_{label}"
        speaker_segments.setdefault(key, []).append(seg)
        speaker_duration[label] = speaker_duration.get(label, 0.0) + (seg[1] - seg[0])

    # Primary speaker = most total speaking time
    primary_label = max(speaker_duration, key=speaker_duration.get)

    flagged_segments: List[Tuple[float, float]] = []
    for label, segs in enumerate(
        [speaker_segments.get(f"speaker_{i}", []) for i in range(num_speakers)]
    ):
        if label != primary_label:
            flagged_segments.extend(segs)

    flagged_segments.sort(key=lambda x: x[0])

    # Confidence = mean intra-cluster cosine similarity
    intra_sims: List[float] = []
    for label in range(num_speakers):
        members = [i for i, lbl in enumerate(labels) if lbl == label]
        if len(members) < 2:
            intra_sims.append(1.0)
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                sim = _cosine_similarity(emb_matrix[members[i]], emb_matrix[members[j]])
                intra_sims.append(sim)

    confidence = float(np.mean(intra_sims)) if intra_sims else 1.0

    if num_speakers > 1:
        non_primary = [f"speaker_{i}" for i in range(num_speakers) if i != primary_label]
        logger.info(
            f"Primary speaker: speaker_{primary_label} "
            f"({speaker_duration[primary_label]:.1f}s). "
            f"Non-primary voice(s): {non_primary}. "
            f"Flagged segments: {len(flagged_segments)}. "
            f"Cluster confidence: {confidence:.3f}"
        )

    return {
        "num_speakers": num_speakers,
        "speaker_segments": speaker_segments,
        "flagged_segments": flagged_segments,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_distinct_voices(
    wav_path: str,
    speech_segments: List[Tuple[float, float]],
) -> Dict[str, Any]:
    """
    Counts the number of distinct voices in an audio file and identifies
    which speech segments belong to the primary speaker vs additional voices.

    Args:
        wav_path:        Path to a 16 kHz mono WAV file.
        speech_segments: List of (start_s, end_s) speech segments from VAD.

    Returns a dict with:
        num_speakers     (int)   — total distinct voices detected
        speaker_segments (dict)  — {speaker_N: [(start, end), ...]}
        flagged_segments (list)  — non-primary speaker segments
        confidence       (float) — clustering quality (0–1)
    """
    if settings.MOCK_ML_MODELS or not wav_path or not os.path.exists(wav_path):
        return {
            "num_speakers": 2,
            "speaker_segments": {
                "speaker_0": [(10.0, 12.0), (60.0, 65.0), (140.0, 142.0)],
                "speaker_1": [(100.0, 105.0)],
            },
            "flagged_segments": [(100.0, 105.0)],
            "confidence": 0.88,
        }

    try:
        return _count_voices_resemblyzer(wav_path, speech_segments)
    except Exception as exc:
        logger.error(f"Resemblyzer voice counting failed: {exc}")
        return {
            "num_speakers": 1,
            "speaker_segments": {},
            "flagged_segments": [],
            "confidence": 0.0,
        }
