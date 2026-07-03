"""
Head Pose / Gaze Estimation Module

Backend priority (GAZE_BACKEND):
  1. 6drepnet  — 6D Rotation Representation Network: direct CNN regression of Euler angles.
                 3.98° MAE on BIWI benchmark vs ~5.5° for PnP methods.
                 ResNet-18 backbone, ~48MB weights, ~35ms/frame CPU.
  2. mediapipe — MediaPipe FaceMesh + solvePnP (decent, 0 extra download required).

6DRepNet weights auto-download from HuggingFace Hub on first use.
"""
import logging
import threading
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 6DRepNet
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False

# ---------------------------------------------------------------------------
# 6DRepNet model definition (lightweight, self-contained)
# ---------------------------------------------------------------------------
_SIXD_MODEL = None
_SIXD_LOCK = threading.Lock()
_SIXD_TRANSFORM = None

# Weights URL (official 6DRepNet checkpoint fine-tuned on 300W-LP + BIWI)
_SIXD_WEIGHTS_URL = (
    "https://huggingface.co/spaces/osanseviero/6DRepNet/resolve/main/"
    "model_weights/6DRepNet_300W_LP_BIWI.pth"
)
_SIXD_WEIGHTS_PATH = Path.home() / ".cache" / "proctoring" / "6DRepNet_300W_LP_BIWI.pth"


def _rotation_matrix_to_euler(R: np.ndarray) -> Dict[str, float]:
    """Converts a 3×3 rotation matrix to yaw/pitch/roll in degrees."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
        roll  = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
        roll  = 0.0
    return {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)}


class _SixDRepNet(nn.Module):
    """
    Minimal 6DRepNet implementation using torchvision ResNet-18 backbone.
    Regresses a 6D rotation representation and decodes to a 3×3 rotation matrix.
    """
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        self.backbone = models.resnet18(weights=None)
        # Replace final FC: 512 → 6 (6D rotation representation)
        self.backbone.fc = nn.Linear(512, 6)

    def forward(self, x):
        out = self.backbone(x)
        # Gram-Schmidt orthonormalization to get rotation matrix
        a1 = out[:, :3]
        a2 = out[:, 3:]
        b1 = nn.functional.normalize(a1, dim=1)
        b2 = nn.functional.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=-1)  # (B, 3, 3)


def _get_sixd_model() -> Optional[Any]:
    """Lazy-load 6DRepNet with auto-download of pretrained weights."""
    global _SIXD_MODEL, _SIXD_TRANSFORM
    if not _HAS_TORCH:
        return None
    if _SIXD_MODEL is None:
        with _SIXD_LOCK:
            if _SIXD_MODEL is None:
                try:
                    _SIXD_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                    if not _SIXD_WEIGHTS_PATH.exists():
                        logger.info(f"Downloading 6DRepNet weights (~48MB)...")
                        import urllib.request
                        urllib.request.urlretrieve(_SIXD_WEIGHTS_URL, _SIXD_WEIGHTS_PATH)
                        logger.info("6DRepNet weights downloaded.")

                    model = _SixDRepNet()
                    state = torch.load(_SIXD_WEIGHTS_PATH, map_location="cpu")
                    # Handle DataParallel-wrapped checkpoints
                    if any(k.startswith("module.") for k in state.keys()):
                        state = {k.replace("module.", ""): v for k, v in state.items()}
                    model.load_state_dict(state, strict=False)
                    model.eval()
                    _SIXD_MODEL = model

                    _SIXD_TRANSFORM = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        ),
                    ])
                    logger.info("6DRepNet loaded successfully.")
                except Exception as exc:
                    logger.error(f"6DRepNet failed to load: {exc}")
    return _SIXD_MODEL


# ---------------------------------------------------------------------------
# MediaPipe FaceMesh (fallback)
# ---------------------------------------------------------------------------
_MP_FACE_MESH = None
_MP_LOCK = threading.Lock()

# 3D face model and landmark indices for PnP
_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0),
], dtype=np.float64)
_LM_INDICES = [1, 152, 263, 33, 287, 57]


def _get_face_mesh():
    global _MP_FACE_MESH
    if not _HAS_MEDIAPIPE:
        return None
    if _MP_FACE_MESH is None:
        with _MP_LOCK:
            if _MP_FACE_MESH is None:
                logger.info("Initializing MediaPipe FaceMesh (gaze fallback)...")
                _MP_FACE_MESH = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=1,
                    refine_landmarks=True, min_detection_confidence=0.5,
                )
    return _MP_FACE_MESH


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_gaze(frame: np.ndarray, face_box: list, timestamp: float = 0.0) -> Dict[str, float]:
    """
    Estimates head pose as yaw/pitch/roll in degrees.
      - Yaw:   left (+) / right (-)
      - Pitch: upward (+) / downward (-)

    Backend 1: 6DRepNet — direct CNN regression, 3.98° MAE on BIWI.
    Backend 2: MediaPipe FaceMesh + solvePnP — decent, no extra models needed.
    """
    if settings.MOCK_ML_MODELS:
        if 50.0 <= timestamp <= 60.0:
            return {"yaw": 35.0, "pitch": 5.0, "roll": 0.0}
        return {"yaw": 2.0, "pitch": -1.0, "roll": 0.0}

    backend = settings.GAZE_BACKEND.lower()

    # --- 6DRepNet ---
    if backend == "6drepnet":
        model = _get_sixd_model()
        if model is not None and _SIXD_TRANSFORM is not None:
            try:
                x1, y1, x2, y2 = face_box
                # Expand crop slightly for better head pose context
                h, w = frame.shape[:2]
                pad = 20
                crop = frame[
                    max(0, y1 - pad): min(h, y2 + pad),
                    max(0, x1 - pad): min(w, x2 + pad),
                ]
                if crop.size == 0:
                    raise ValueError("Empty face crop")

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                inp = _SIXD_TRANSFORM(crop_rgb).unsqueeze(0)

                with torch.no_grad():
                    R = model(inp)[0].numpy()  # (3, 3) rotation matrix

                return _rotation_matrix_to_euler(R)
            except Exception as exc:
                logger.debug(f"6DRepNet failed at t={timestamp:.1f}s: {exc}")

    # --- MediaPipe FaceMesh + solvePnP ---
    face_mesh = _get_face_mesh()
    if face_mesh is not None:
        try:
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                image_points = np.array(
                    [(lm[i].x * w, lm[i].y * h) for i in _LM_INDICES],
                    dtype=np.float64,
                )
                focal = w
                cam = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=np.float64)
                ok, rvec, _ = cv2.solvePnP(_MODEL_POINTS, image_points, cam, np.zeros((4,1)))
                if ok:
                    rmat, _ = cv2.Rodrigues(rvec)
                    return _rotation_matrix_to_euler(rmat)
        except Exception as exc:
            logger.debug(f"MediaPipe gaze failed at t={timestamp:.1f}s: {exc}")

    return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}


# Allow Any type for model annotation
from typing import Any
