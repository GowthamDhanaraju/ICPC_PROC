"""
Speaker Diarization / Voice Counting Module

Upgrade: binary "second speaker detected" → multi-speaker voice count

Instead of simply asking "is there a second voice?", this module now answers:
  "How many distinct voices are present, and when does each one speak?"

Algorithm (resemblyzer backend):
  1. Run all detected speech segments through the GE2E speaker encoder,
     producing a 256-dim embedding per segment.
  2. Build a cosine-distance matrix across all embeddings.
  3. Apply agglomerative clustering (single-linkage, threshold=0.25) to group
     embeddings into speaker clusters.  Each cluster = one distinct voice.
  4. The primary speaker is the cluster with the most total speaking time.
     All other clusters are returned as "flagged" (non-primary voice) segments.
  5. Confidence = mean intra-cluster similarity (how tightly each speaker
     cluster is packed), averaged across all clusters.

Backends:
  - resemblyzer (default) — GE2E speaker encoder, 17 MB, no token required.
  - pyannote (optional)   — pyannote/speaker-diarization-3.1, state-of-the-art,
                            requires HUGGINGFACE_TOKEN.

Enable pyannote by setting DIARIZATION_BACKEND=pyannote + HUGGINGFACE_TOKEN in .env.
"""
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# resemblyzer backend
# ---------------------------------------------------------------------------
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
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
# Agglomerative clustering helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised (or any) vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def _agglomerative_cluster(
    embeddings: np.ndarray,
    distance_threshold: float = 0.25,
) -> List[int]:
    """
    Simple single-linkage agglomerative clustering on cosine distance.

    Works without scikit-learn by building a distance matrix manually.
    distance_threshold = 1 - similarity_threshold
      → 0.25 corresponds to similarity ≥ 0.75 (same speaker)

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

    # Single-linkage merging
    labels = list(range(n))  # each point starts in its own cluster

    def _min_dist_between_clusters(c1: int, c2: int) -> float:
        members_c1 = [k for k, lbl in enumerate(labels) if lbl == c1]
        members_c2 = [k for k, lbl in enumerate(labels) if lbl == c2]
        return float(min(dist[i, j] for i in members_c1 for j in members_c2))

    while True:
        # Find the pair of distinct clusters with minimum inter-cluster distance
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
            break  # Closest clusters are still too far apart — stop merging

        # Merge cj into ci
        ci, cj = best_pair
        labels = [ci if lbl == cj else lbl for lbl in labels]

    # Re-index labels to 0, 1, 2, ...
    unique = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique)}
    return [remap[lbl] for lbl in labels]


# ---------------------------------------------------------------------------
# resemblyzer voice counting
# ---------------------------------------------------------------------------

def _count_voices_resemblyzer(
    wav_path: str,
    speech_segments: List[Tuple[float, float]],
) -> Dict[str, Any]:
    """
    Embeds all speech segments and clusters them into distinct speaker groups.

    Returns:
        num_speakers  — count of distinct voices (int)
        speaker_segments — dict mapping "speaker_N" → list of (start, end) tuples
        flagged_segments  — segments NOT belonging to the primary (most-speaking) speaker
        confidence    — mean intra-cluster cosine similarity (0–1)
    """
    encoder = _get_resemblyzer_encoder()
    if encoder is None:
        raise RuntimeError("resemblyzer not available")

    # Need at least 2 segments to compare
    if len(speech_segments) < 2:
        return {
            "num_speakers": 1,
            "speaker_segments": {"speaker_0": list(speech_segments)},
            "flagged_segments": [],
            "confidence": 1.0,
        }

    # --- Load audio ---
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

    # --- Embed each segment ---
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

    # --- Cluster embeddings into speaker groups ---
    # distance_threshold=0.25  ↔  similarity ≥ 0.75 → same speaker
    labels = _agglomerative_cluster(emb_matrix, distance_threshold=0.25)
    num_speakers = max(labels) + 1

    logger.info(
        f"Voice counting: {len(valid_segments)} segments → {num_speakers} distinct speaker(s)"
    )

    # --- Build per-speaker segment lists ---
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

    # --- Compute confidence (mean intra-cluster similarity) ---
    intra_sims: List[float] = []
    for label in range(num_speakers):
        members = [i for i, lbl in enumerate(labels) if lbl == label]
        if len(members) < 2:
            intra_sims.append(1.0)  # single-member cluster: perfect similarity
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                sim = _cosine_similarity(emb_matrix[members[i]], emb_matrix[members[j]])
                intra_sims.append(sim)

    confidence = float(np.mean(intra_sims)) if intra_sims else 1.0

    if num_speakers > 1:
        non_primary = [
            f"speaker_{i}" for i in range(num_speakers) if i != primary_label
        ]
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
# pyannote voice counting
# ---------------------------------------------------------------------------

def _count_voices_pyannote(wav_path: str) -> Dict[str, Any]:
    """
    Runs pyannote/speaker-diarization-3.1 and returns a voice-count result
    in the same format as _count_voices_resemblyzer.
    Requires HUGGINGFACE_TOKEN to be set.
    """
    pipeline = _get_pyannote_pipeline()
    if pipeline is None:
        raise RuntimeError("pyannote pipeline not available (check HUGGINGFACE_TOKEN)")

    diarization = pipeline(wav_path)

    from collections import defaultdict
    speaker_segs: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    speaker_duration: Dict[str, float] = {}

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segs[speaker].append((turn.start, turn.end))
        dur = turn.end - turn.start
        speaker_duration[speaker] = speaker_duration.get(speaker, 0.0) + dur

    num_speakers = len(speaker_segs)
    if num_speakers == 0:
        return {
            "num_speakers": 0,
            "speaker_segments": {},
            "flagged_segments": [],
            "confidence": 1.0,
        }

    # Primary = most total speaking time
    primary = max(speaker_duration, key=speaker_duration.get)

    flagged: List[Tuple[float, float]] = []
    for spk, segs in speaker_segs.items():
        if spk != primary:
            flagged.extend(segs)

    flagged.sort(key=lambda x: x[0])

    # Rename to speaker_0, speaker_1, ... for consistent output format
    label_map = {spk: f"speaker_{i}" for i, spk in enumerate(sorted(speaker_segs))}
    formatted_segments = {label_map[spk]: segs for spk, segs in speaker_segs.items()}

    logger.info(
        f"pyannote: {num_speakers} distinct speaker(s) detected. "
        f"Primary: {primary}. Flagged segments: {len(flagged)}."
    )

    return {
        "num_speakers": num_speakers,
        "speaker_segments": formatted_segments,
        "flagged_segments": flagged,
        "confidence": 0.92,  # pyannote doesn't expose per-cluster similarity; use model-level accuracy
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
    which speech segments belong to the primary (exam candidate) speaker vs
    additional voices (potential collaborators / second people in the room).

    Args:
        wav_path:        Path to a 16 kHz mono WAV file.
        speech_segments: List of (start_s, end_s) speech segments from VAD.

    Returns a dict with:
        num_speakers     (int)   — total distinct voices detected
        speaker_segments (dict)  — {speaker_N: [(start, end), ...]}
        flagged_segments (list)  — non-primary speaker segments (potential violations)
        confidence       (float) — clustering quality / model confidence (0–1)

    Backend cascade: pyannote (if token set) → resemblyzer → mock/empty
    """
    if settings.MOCK_ML_MODELS or not wav_path or not os.path.exists(wav_path):
        # Mock: simulate 2 speakers with one flagged segment
        return {
            "num_speakers": 2,
            "speaker_segments": {
                "speaker_0": [(10.0, 12.0), (60.0, 65.0), (140.0, 142.0)],
                "speaker_1": [(100.0, 105.0)],
            },
            "flagged_segments": [(100.0, 105.0)],
            "confidence": 0.88,
        }

    backend = settings.DIARIZATION_BACKEND.lower()

    # --- pyannote (best accuracy, requires HuggingFace token) ---
    if backend == "pyannote" and settings.HUGGINGFACE_TOKEN:
        try:
            return _count_voices_pyannote(wav_path)
        except Exception as exc:
            logger.warning(
                f"pyannote voice counting failed: {exc}. Falling back to resemblyzer."
            )

    # --- resemblyzer GE2E (good accuracy, no token needed) ---
    if backend in ("resemblyzer", "pyannote"):
        try:
            return _count_voices_resemblyzer(wav_path, speech_segments)
        except Exception as exc:
            logger.warning(f"resemblyzer voice counting failed: {exc}.")

    logger.warning("No diarization backend available — voice counting skipped.")
    return {
        "num_speakers": 1,
        "speaker_segments": {},
        "flagged_segments": [],
        "confidence": 0.0,
    }
