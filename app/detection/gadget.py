"""
Prohibited gadget / object detection module.

Thread safety: Model singleton protected by threading.Lock using
double-checked locking pattern, same as face.py.
"""
import threading
import logging
import numpy as np
from typing import List, Dict, Any
from app.config import settings

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False

_GADGET_MODEL = None
_GADGET_MODEL_LOCK = threading.Lock()

# Prohibited object class names (from COCO or custom fine-tuned model)
PROHIBITED_CLASSES = {"cell phone", "book", "laptop", "remote"}


def get_gadget_model():
    """
    Returns a lazily-initialized, thread-safe gadget detection model singleton.
    """
    global _GADGET_MODEL
    if settings.MOCK_ML_MODELS or not _HAS_YOLO:
        return None
    if _GADGET_MODEL is None:
        with _GADGET_MODEL_LOCK:
            if _GADGET_MODEL is None:
                logger.info("Loading gadget detection model (yolov8n.pt)...")
                try:
                    _GADGET_MODEL = YOLO("yolov8n.pt")
                except Exception as e:
                    logger.error(f"Failed to load gadget detection model: {e}")
    return _GADGET_MODEL


def detect_gadgets(frame: np.ndarray, timestamp: float = 0.0) -> List[Dict[str, Any]]:
    """
    Detects prohibited gadgets/objects in the frame.
    Returns a list of dicts:
      - box: [x1, y1, x2, y2]
      - class_name: str
      - confidence: float
    """
    if settings.MOCK_ML_MODELS:
        # Mock timeline: 80s–85s → cell phone present
        if 80.0 <= timestamp <= 85.0:
            return [{"box": [50, 200, 150, 350], "class_name": "cell phone", "confidence": 0.89}]
        return []

    model = get_gadget_model()
    if not model:
        return []

    results = model(frame, verbose=False)
    detections = []
    for r in results:
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = names.get(cls_id, "")
            conf = float(box.conf[0])
            if class_name in PROHIBITED_CLASSES and conf >= settings.GADGET_CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "class_name": class_name,
                    "confidence": conf,
                })
    return detections
