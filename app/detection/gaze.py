"""
Gaze / head pose estimation module.

Estimates yaw (left/right) and pitch (up/down) rotation in degrees
by solving the PnP problem against a 3D facial landmark model via MediaPipe.
"""
import cv2
import threading
import logging
import numpy as np
from typing import Dict
from app.config import settings

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False

_MP_FACE_MESH = None
_MP_LOCK = threading.Lock()


def get_face_mesh():
    """
    Returns a lazily-initialized, thread-safe MediaPipe FaceMesh instance.
    """
    global _MP_FACE_MESH
    if settings.MOCK_ML_MODELS or not _HAS_MEDIAPIPE:
        return None
    if _MP_FACE_MESH is None:
        with _MP_LOCK:
            if _MP_FACE_MESH is None:
                logger.info("Initializing MediaPipe FaceMesh for gaze estimation...")
                _MP_FACE_MESH = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                )
    return _MP_FACE_MESH


# 3D model points for 6 canonical face landmarks (generic model)
_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye corner
    (225.0, 170.0, -135.0),   # Right eye corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

# MediaPipe FaceMesh indices for the 6 landmark points above
_LM_INDICES = [1, 152, 263, 33, 287, 57]


def estimate_gaze(frame: np.ndarray, face_box: list, timestamp: float = 0.0) -> Dict[str, float]:
    """
    Estimates gaze / head pose as pitch and yaw in degrees.
      - Yaw:   left (+) / right (-)
      - Pitch: upward (+) / downward (-)
    """
    if settings.MOCK_ML_MODELS:
        # Mock: 50s–60s → sustained gaze deviation
        if 50.0 <= timestamp <= 60.0:
            return {"yaw": 35.0, "pitch": 5.0}
        return {"yaw": 2.0, "pitch": -1.0}

    face_mesh = get_face_mesh()
    if not face_mesh:
        return {"yaw": 0.0, "pitch": 0.0}

    try:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {"yaw": 0.0, "pitch": 0.0}

        lm = results.multi_face_landmarks[0].landmark
        image_points = np.array([
            (lm[i].x * w, lm[i].y * h) for i in _LM_INDICES
        ], dtype=np.float64)

        # Camera intrinsics (approximate)
        focal = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal, 0, center[0]],
            [0, focal, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, _ = cv2.solvePnP(
            _MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return {"yaw": 0.0, "pitch": 0.0}

        rmat, _ = cv2.Rodrigues(rvec)
        proj = np.hstack((rmat, np.zeros((3, 1))))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
        pitch = float(euler[0, 0])
        yaw = float(euler[1, 0])
        return {"yaw": yaw, "pitch": pitch}

    except Exception as e:
        logger.debug(f"Gaze estimation failed at t={timestamp:.1f}s: {e}")
        return {"yaw": 0.0, "pitch": 0.0}
