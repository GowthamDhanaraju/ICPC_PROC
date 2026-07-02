from typing import List, Dict, Any, Tuple
from app.config import settings

class EventAggregator:
    def __init__(self):
        # Configuration for merging and debouncing
        self.merge_windows = {
            "MULTIPLE_FACES": 5.0,        # merge events within 5 seconds
            "NO_FACE_DETECTED": 5.0,
            "GAZE_AWAY_SUSTAINED": 4.0,
            "PROHIBITED_DEVICE": 6.0,
            "SECOND_VOICE_DETECTED": 5.0
        }
        
        self.min_durations = {
            "MULTIPLE_FACES": 1.5,        # Must last at least 1.5s
            "NO_FACE_DETECTED": 2.0,       # Must last at least 2s
            "GAZE_AWAY_SUSTAINED": 3.0,    # Must last at least 3s
            "PROHIBITED_DEVICE": 1.0,      # Must last at least 1s
            "SECOND_VOICE_DETECTED": 1.5   # Must last at least 1.5s
        }

    def aggregate(
        self, 
        frame_detections: List[Dict[str, Any]], 
        audio_detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Aggregates frame detections and audio detections into a final list of violations.
        - frame_detections: List of dicts, each having:
            {
              "timestamp": float,
              "faces": List[dict],
              "gaze": {"pitch": float, "yaw": float},
              "gadgets": List[dict]
            }
        - audio_detections: List of dicts, each having:
            {
              "type": "SECOND_VOICE_DETECTED",
              "start_ts": float,
              "end_ts": float,
              "confidence": float
            }
        """
        violations = []

        # Extract raw flag events from video frames
        video_flags = {
            "MULTIPLE_FACES": [],
            "NO_FACE_DETECTED": [],
            "GAZE_AWAY_SUSTAINED": [],
            "PROHIBITED_DEVICE": []
        }

        for fd in frame_detections:
            t = fd["timestamp"]
            faces = fd.get("faces", [])
            gaze = fd.get("gaze", {"yaw": 0.0, "pitch": 0.0})
            gadgets = fd.get("gadgets", [])

            # MULTIPLE_FACES
            if len(faces) >= 2:
                # Average face confidence
                conf = sum(f["confidence"] for f in faces) / len(faces)
                video_flags["MULTIPLE_FACES"].append((t, conf))

            # NO_FACE_DETECTED
            elif len(faces) == 0:
                video_flags["NO_FACE_DETECTED"].append((t, 1.0))  # 100% confident no face seen

            # GAZE_AWAY_SUSTAINED
            yaw = abs(gaze.get("yaw", 0.0))
            pitch = abs(gaze.get("pitch", 0.0))
            if yaw > settings.GAZE_AWAY_YAW_THRESHOLD or pitch > settings.GAZE_AWAY_PITCH_THRESHOLD:
                video_flags["GAZE_AWAY_SUSTAINED"].append((t, 0.8))

            # PROHIBITED_DEVICE
            if len(gadgets) > 0:
                conf = max(g["confidence"] for g in gadgets)
                video_flags["PROHIBITED_DEVICE"].append((t, conf))

        # Process each video flag type into contiguous intervals
        for vtype, points in video_flags.items():
            if not points:
                continue
            
            # Sort points by timestamp
            points = sorted(points, key=lambda x: x[0])
            
            # Group points into candidate intervals using the merge window
            intervals: List[List[Tuple[float, float]]] = []
            merge_win = self.merge_windows[vtype]
            
            for pt in points:
                t, conf = pt
                if not intervals or t - intervals[-1][-1][0] > merge_win:
                    intervals.append([pt])
                else:
                    intervals[-1].append(pt)
            
            # Convert intervals to violation events if they pass debounce (min duration)
            min_dur = self.min_durations[vtype]
            for group in intervals:
                start_t = group[0][0]
                end_t = group[-1][0]
                duration = end_t - start_t
                
                # Debounce: if the event duration is longer than the minimum required
                if duration >= min_dur or (vtype == "PROHIBITED_DEVICE" and len(group) >= 1):
                    # For a single frame of gadget, duration might be 0, but it is still a violation.
                    # We ensure duration is at least 1.0s or baseline interval
                    if duration == 0:
                        duration = settings.ADAPTIVE_SAMPLING_BASELINE_INTERVAL
                        end_t = start_t + duration
                    
                    avg_conf = sum(pt[1] for pt in group) / len(group)
                    violations.append({
                        "type": vtype,
                        "start_ts": start_t,
                        "end_ts": end_t,
                        "duration": duration,
                        "confidence": avg_conf
                    })

        # Process audio violations
        audio_groups: Dict[str, List[Dict[str, Any]]] = {}
        for ad in audio_detections:
            atype = ad["type"]
            audio_groups.setdefault(atype, []).append(ad)

        for atype, events in audio_groups.items():
            events = sorted(events, key=lambda x: x["start_ts"])
            merge_win = self.merge_windows.get(atype, 5.0)
            
            merged_events = []
            for ev in events:
                if not merged_events or ev["start_ts"] - merged_events[-1]["end_ts"] > merge_win:
                    merged_events.append(ev.copy())
                else:
                    # Merge with previous
                    merged_events[-1]["end_ts"] = max(merged_events[-1]["end_ts"], ev["end_ts"])
                    merged_events[-1]["duration"] = merged_events[-1]["end_ts"] - merged_events[-1]["start_ts"]
                    merged_events[-1]["confidence"] = (merged_events[-1]["confidence"] + ev["confidence"]) / 2.0

            # Filter merged events by min duration
            min_dur = self.min_durations.get(atype, 1.5)
            for ev in merged_events:
                if ev["duration"] >= min_dur:
                    violations.append({
                        "type": atype,
                        "start_ts": ev["start_ts"],
                        "end_ts": ev["end_ts"],
                        "duration": ev["duration"],
                        "confidence": ev["confidence"]
                    })

        # Sort all final violations by start timestamp
        violations.sort(key=lambda x: x["start_ts"])
        return violations
