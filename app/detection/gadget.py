import cv2
import numpy as np
from typing import List, Dict, Any
from app.config import settings

# Attempt actual YOLO import
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False

_GADGET_MODEL = None

def get_gadget_model():
    global _GADGET_MODEL
    if not settings.MOCK_ML_MODELS and _HAS_YOLO:
        if _GADGET_MODEL is None:
            # Load fine-tuned YOLO model for gadgets
            try:
                _GADGET_MODEL = YOLO("yolov8n.pt")  # Fallback to general yolov8n
            except Exception:
                pass
    return _GADGET_MODEL

# Prohibited object classes in COCO or custom fine-tuned model
PROHIBITED_CLASSES = {"cell phone", "book", "laptop", "remote"}

def detect_gadgets(frame: np.ndarray, timestamp: float = 0.0) -> List[Dict[str, Any]]:
    """
    Detects prohibited gadgets/objects in the frame.
    Returns a list of dicts:
      - box: [x1, y1, x2, y2]
      - class_name: str ("cell phone", "book", "laptop", etc.)
      - confidence: float
    """
    if settings.MOCK_ML_MODELS:
        # Mock gadget timeline:
        # 80s to 85s: "cell phone" in frame
        # Otherwise: None
        if 80.0 <= timestamp <= 85.0:
            return [
                {"box": [50, 200, 150, 350], "class_name": "cell phone", "confidence": 0.89}
            ]
        return []

    # Real YOLOv8 detection logic
    model = get_gadget_model()
    if not model:
        return []

    results = model(frame, verbose=False)
    detections = []
    
    for r in results:
        # Get names dictionary from model
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
                    "confidence": conf
                })
                
    return detections
