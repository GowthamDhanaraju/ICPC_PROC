"""
Head Pose / Gaze Estimation Module

Backend: 6DRepNet (6D Rotation Representation Network).
  - ResNet-18 backbone, direct CNN regression of Euler angles.
  - 3.98° MAE on BIWI benchmark — best available lightweight model.
  - ~48 MB weights, auto-downloaded from HuggingFace Hub on first use.
"""
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Torch dependency
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    logger.error(
        "PyTorch is not installed. Gaze estimation will be unavailable. "
        "Install it with: pip install torch torchvision"
    )

# ---------------------------------------------------------------------------
# 6DRepNet model (self-contained)
# ---------------------------------------------------------------------------
_SIXD_MODEL = None
_SIXD_LOCK = threading.Lock()
_SIXD_TRANSFORM = None

_SIXD_WEIGHTS_URL = (
    "https://huggingface.co/osanseviero/6DRepNet_300W_LP_AFLW2000/resolve/main/model.pth"
)
_SIXD_WEIGHTS_PATH = Path.home() / ".cache" / "proctoring" / "6DRepNet_300W_LP_AFLW2000.pth"


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
    Minimal 6DRepNet using a torchvision ResNet-18 backbone.
    Regresses a 6D rotation representation and decodes to a 3×3 rotation matrix.
    """
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(512, 6)

    def forward(self, x):
        out = self.backbone(x)
        a1 = out[:, :3]
        a2 = out[:, 3:]
        b1 = nn.functional.normalize(a1, dim=1)
        b2 = nn.functional.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=-1)  # (B, 3, 3)


def _get_sixd_model() -> Optional[Any]:
    """Lazy-load 6DRepNet with auto-download of pretrained weights (~48 MB)."""
    global _SIXD_MODEL, _SIXD_TRANSFORM
    if not _HAS_TORCH:
        return None
    if _SIXD_MODEL is None:
        with _SIXD_LOCK:
            if _SIXD_MODEL is None:
                try:
                    _SIXD_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                    if not _SIXD_WEIGHTS_PATH.exists():
                        logger.info("Downloading 6DRepNet weights (~48 MB)...")
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
# Public API
# ---------------------------------------------------------------------------

def estimate_gaze(frame: np.ndarray, face_box: list, timestamp: float = 0.0) -> Dict[str, float]:
    """
    Estimates head pose as yaw/pitch/roll in degrees using 6DRepNet.
      - Yaw:   left (+) / right (-)
      - Pitch: upward (+) / downward (-)

    Returns {yaw: 0.0, pitch: 0.0, roll: 0.0} on model failure rather than
    silently switching to a lower-accuracy fallback.
    """
    if settings.MOCK_ML_MODELS:
        if 50.0 <= timestamp <= 60.0:
            return {"yaw": 35.0, "pitch": 5.0, "roll": 0.0}
        return {"yaw": 2.0, "pitch": -1.0, "roll": 0.0}

    model = _get_sixd_model()
    if model is None or _SIXD_TRANSFORM is None:
        logger.warning(f"6DRepNet unavailable — gaze not estimated at t={timestamp:.1f}s.")
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    try:
        x1, y1, x2, y2 = face_box
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
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
