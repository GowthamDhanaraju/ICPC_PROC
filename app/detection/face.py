"""
Face Detection and Identity Verification Module

Backend: InsightFace SCRFD (detector) + ArcFace buffalo_sc (identity).
  - SCRFD: 92.8 mAP on WiderFace Hard — best single-model CPU-capable detector.
  - ArcFace buffalo_sc: 99.5% LFW accuracy — best lightweight embedding model.

All singletons use threading.Lock for thread-safe lazy initialization.
"""
import logging
import os
import threading
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# InsightFace (SCRFD + ArcFace) — sole backend
# ---------------------------------------------------------------------------
try:
    import insightface
    from insightface.app import FaceAnalysis
    _HAS_INSIGHTFACE = True
except ImportError:
    _HAS_INSIGHTFACE = False
    logger.error(
        "InsightFace is not installed. Face detection and identity verification "
        "will be unavailable. Install it with: pip install insightface"
    )

_INSIGHT_APP: Optional[Any] = None
_INSIGHT_LOCK = threading.Lock()


def _get_insightface_app() -> Optional[Any]:
    """
    Lazy-init InsightFace FaceAnalysis with SCRFD detector + ArcFace recognition.
    Model 'buffalo_sc' (small+fast): ~300 MB, auto-downloads on first use.
    """
    global _INSIGHT_APP
    if not _HAS_INSIGHTFACE:
        return None
    if _INSIGHT_APP is None:
        with _INSIGHT_LOCK:
            if _INSIGHT_APP is None:
                logger.info("Loading InsightFace FaceAnalysis (buffalo_sc)...")
                try:
                    app = FaceAnalysis(
                        name="buffalo_sc",
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    )
                    app.prepare(ctx_id=settings.INSIGHTFACE_CTX_ID, det_size=(640, 640))
                    _INSIGHT_APP = app
                    logger.info("InsightFace loaded successfully.")
                except Exception as exc:
                    logger.error(f"InsightFace failed to load: {exc}")
    return _INSIGHT_APP


# ---------------------------------------------------------------------------
# Public API — Face Detection
# ---------------------------------------------------------------------------

def detect_faces(frame: np.ndarray, timestamp: float = 0.0) -> List[Dict[str, Any]]:
    """
    Detects faces in a video frame using InsightFace SCRFD.
    Returns list of {'box': [x1,y1,x2,y2], 'confidence': float}.

    Raises a warning and returns [] if the model is unavailable rather than
    silently falling back to a lower-accuracy detector.
    """
    if settings.MOCK_ML_MODELS:
        if 10.0 <= timestamp <= 15.0:
            return [
                {"box": [100, 100, 200, 200], "confidence": 0.95},
                {"box": [300, 100, 400, 200], "confidence": 0.91},
            ]
        elif 30.0 <= timestamp <= 35.0:
            return []
        return [{"box": [200, 100, 300, 200], "confidence": 0.96}]

    app = _get_insightface_app()
    if app is None:
        logger.warning(f"InsightFace unavailable — no face detection at t={timestamp:.1f}s.")
        return []

    try:
        faces = app.get(frame)
        results = []
        for face in faces:
            if face.det_score >= settings.FACE_CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, face.bbox)
                results.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": float(face.det_score),
                    "_insightface_obj": face,  # kept for identity reuse
                })
        return results
    except Exception as exc:
        logger.warning(f"InsightFace inference failed at t={timestamp:.1f}s: {exc}")
        return []


# ---------------------------------------------------------------------------
# Public API — Identity Verification
# ---------------------------------------------------------------------------

def verify_identity(
    face_crop: np.ndarray,
    enrollment_photo_uri: str,
    candidate_id: str,
    temp_dir: Optional[str] = None,
) -> float:
    """
    Compares a live face crop against an enrollment photo using ArcFace embeddings.
    Returns cosine similarity in [0.0, 1.0]. Higher = more similar.

    A mismatch is flagged in the worker when similarity < 0.50 (strict).

    temp_dir — when provided, the enrollment photo is downloaded there and
    reused on subsequent calls within the same job (per-job download cache).
    """
    if settings.MOCK_ML_MODELS:
        return 0.15 if candidate_id == "impersonator" else 0.92

    app = _get_insightface_app()
    if app is None:
        logger.warning("ArcFace unavailable — identity verification skipped.")
        return 0.5  # uncertain; not flagged

    try:
        from app.preprocessing.media import get_local_path
        enrollment_path = get_local_path(enrollment_photo_uri, temp_dir)
        enrollment_img = cv2.imread(enrollment_path)
        if enrollment_img is None:
            raise FileNotFoundError(f"Enrollment photo not readable: {enrollment_path}")

        live_faces   = app.get(face_crop)
        enroll_faces = app.get(enrollment_img)

        if not live_faces or not enroll_faces:
            logger.warning("No face detected in crop or enrollment photo.")
            return 0.5  # uncertain

        emb_live   = live_faces[0].normed_embedding
        emb_enroll = enroll_faces[0].normed_embedding

        # Cosine similarity (both are already L2-normalised by InsightFace)
        similarity = float(np.dot(emb_live, emb_enroll))
        return max(0.0, min(1.0, similarity))
    except Exception as exc:
        logger.warning(f"ArcFace identity verification failed: {exc}")
        return 0.5  # uncertain; not flagged


def get_face_embedding(face_crop: np.ndarray) -> Optional[np.ndarray]:
    """
    Returns a 512-dim ArcFace embedding for a face crop, or None on failure.
    Used by the diarization module to cluster face tracks.
    """
    app = _get_insightface_app()
    if app is None:
        return None
    try:
        faces = app.get(face_crop)
        if faces:
            return faces[0].normed_embedding
    except Exception:
        pass
    return None
