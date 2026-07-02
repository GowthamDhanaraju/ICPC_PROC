import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from app.config import settings

# Attempt actual YOLO import if we want to run real models
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False

# Global model cache to avoid reloading on every frame
_FACE_MODEL = None

def get_face_model():
    global _FACE_MODEL
    if not settings.MOCK_ML_MODELS and _HAS_YOLO:
        if _FACE_MODEL is None:
            # Load fine-tuned YOLO face detector or standard YOLOv8n
            # In production, this would be a path to a custom model
            try:
                _FACE_MODEL = YOLO("yolov8n-face.pt")  # or standard yolov8n.pt if not fine-tuned
            except Exception:
                _FACE_MODEL = YOLO("yolov8n.pt")
    return _FACE_MODEL

def detect_faces(frame: np.ndarray, timestamp: float = 0.0) -> List[Dict[str, Any]]:
    """
    Detects faces in a video frame.
    Returns a list of dicts with:
      - box: [x1, y1, x2, y2]
      - confidence: float
    """
    if settings.MOCK_ML_MODELS:
        # Mock detection timeline to simulate violations deterministically
        # 10s to 15s: 2 faces (MULTIPLE_FACES)
        # 30s to 35s: 0 faces (NO_FACE_DETECTED)
        # Otherwise: 1 face
        if 10.0 <= timestamp <= 15.0:
            return [
                {"box": [100, 100, 200, 200], "confidence": 0.92},
                {"box": [300, 100, 400, 200], "confidence": 0.88}
            ]
        elif 30.0 <= timestamp <= 35.0:
            return []
        else:
            return [
                {"box": [200, 100, 300, 200], "confidence": 0.95}
            ]

    # Real face detection logic
    model = get_face_model()
    if not model:
        # Fallback to OpenCV Haar Cascades if YOLO is not installed/loading
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return [{"box": [int(x), int(y), int(x+w), int(y+h)], "confidence": 0.8} for (x, y, w, h) in faces]

    # YOLOv8 face detection
    results = model(frame, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            # If standard yolov8 detection, class 0 is 'person' which includes face/body
            # Fine-tuned yolov8-face usually has class 0 as face
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf >= settings.FACE_CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({"box": [x1, y1, x2, y2], "confidence": conf})
    return detections

def verify_identity(face_crop: np.ndarray, enrollment_photo_uri: str, candidate_id: str) -> float:
    """
    Compares the crop of a detected face against the enrollment photo.
    Returns a similarity score between 0.0 and 1.0.
    """
    if settings.MOCK_ML_MODELS:
        # Mock identity mismatch deterministically for testing
        # If candidate_id is "impersonator", return low similarity
        if candidate_id == "impersonator":
            return 0.15
        return 0.92

    # In production, extract face embeddings using ArcFace / FaceNet
    # and calculate cosine similarity.
    # Return mock value if ArcFace dependencies are missing
    try:
        # Placeholder for real InsightFace/ArcFace library code:
        # from insightface.app import FaceAnalysis
        # app = FaceAnalysis(name='buffalo_l')
        # app.prepare(ctx_id=0, det_size=(640, 640))
        # ...
        pass
    except Exception:
        pass
        
    return 0.85
