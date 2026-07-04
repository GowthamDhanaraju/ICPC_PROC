import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

def render_overlay_video(
    frames_list: List[Tuple[float, np.ndarray]],
    frame_detections: List[Dict[str, Any]],
    output_path: str,
) -> bool:
    """
    Writes a new MP4 video to output_path at 24 FPS directly from the sampled frames.
    It draws the closest bounding boxes and gaze lines from frame_detections.
    """
    logger.info(f"Rendering overlay video to {output_path}...")

    if not frames_list:
        logger.error("No frames to render for overlay.")
        return False

    # Get dimensions from the first frame
    first_frame = frames_list[0][1]
    height, width = first_frame.shape[:2]
    
    fps = 24.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create a mapping from timestamp to detection for fast lookup
    det_map = {d["timestamp"]: d for d in frame_detections}
    
    for ts, frame in frames_list:
        # Create a copy so we don't modify the original frame
        display_frame = frame.copy()
        
        detection = det_map.get(ts)
        if detection:
            _draw_detection(display_frame, detection)
            
        # Draw timestamp overlay
        cv2.putText(
            display_frame, 
            f"Time: {ts:.1f}s | FPS: {fps:.1f}", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, (255, 255, 255), 2
        )

        out.write(display_frame)

    out.release()
    
    logger.info(f"Overlay video rendering complete: {output_path} ({len(frames_list)} frames at {fps} fps)")
    return True

def _draw_detection(frame: np.ndarray, detection: Dict[str, Any]):
    """Draws faces, gaze, and gadgets on the frame."""
    faces = detection.get("faces", [])
    gadgets = detection.get("gadgets", [])
    gaze = detection.get("gaze", {})
    identity_mismatch = detection.get("identity_mismatch")
    
    # 1. Draw Faces
    for i, f in enumerate(faces):
        # x1, y1, x2, y2 from detection model
        box = f.get("box", [])
        if len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
            
            # Color logic: Red if mismatch, otherwise Green
            color = (0, 0, 255) if (i == 0 and identity_mismatch is not None) else (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            label = "Face"
            if i == 0 and identity_mismatch is not None:
                label = f"MISMATCH! ({identity_mismatch:.2f})"
            elif i == 0:
                label = "Primary Face"
            
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw Gaze text only on the primary face
            if i == 0 and gaze:
                yaw = gaze.get("yaw", 0.0)
                pitch = gaze.get("pitch", 0.0)
                
                # Draw Gaze Line projecting from center of face (approx eye level)
                import math
                center_x = (x1 + x2) // 2
                center_y = int(y1 + (y2 - y1) * 0.4)
                
                length = 150.0
                # In standard image coords, positive X is right, positive Y is down.
                # Yaw > 0 is looking left from the subject's perspective (which is to the right in the image).
                # Pitch > 0 is looking down (positive Y).
                dx = int(-math.sin(yaw * math.pi / 180.0) * length)
                dy = int(math.sin(pitch * math.pi / 180.0) * length)
                
                # Draw the arrowed line in red
                cv2.arrowedLine(frame, (center_x, center_y), (center_x + dx, center_y + dy), (0, 0, 255), 4, tipLength=0.2)
                # Draw a small dot at the origin (eyes)
                cv2.circle(frame, (center_x, center_y), 4, (0, 255, 255), -1)
                
                if yaw > 10.0:
                    direction = "Left"
                elif yaw < -10.0:
                    direction = "Right"
                else:
                    direction = "Straight"
                    
                gaze_text = f"Gaze: {direction} (Y:{yaw:.1f} P:{pitch:.1f})"
                cv2.putText(frame, gaze_text, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
    # 2. Draw Gadgets
    for g in gadgets:
        box = g.get("box", [])
        if len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
            label = g.get("label", "unknown")
            conf = g.get("confidence", 0.0)
            
            # Yellow for gadgets
            color = (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            text = f"{label} ({conf:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
