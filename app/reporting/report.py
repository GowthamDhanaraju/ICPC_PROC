"""
Pipeline Report Builder

Assembles raw detection data from all detectors into a structured JSON report.
No scoring — this is a pure observation log with the following top-level sections:

  meta              — job / video metadata
  summary           — high-level counts per signal type
  events            — consolidated time windows (debounced, merged)
  face_timeline     — per-frame face counts, bounding boxes and confidence
  gaze_timeline     — per-frame yaw / pitch / roll measurements
  gadget_timeline   — per-frame prohibited object detections
  identity_timeline — identity-verification results per frame (if enrollment photo provided)
  audio             — VAD speech segments + speaker diarization result

All timestamps are in seconds from the start of the video.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_windows(
    flag_points: List[Tuple[float, float]],
    merge_gap: float = 5.0,
    min_duration: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Merge a list of (timestamp, confidence) points into contiguous time windows.

    Args:
        flag_points:  Sorted list of (ts_seconds, confidence) tuples.
        merge_gap:    If two consecutive timestamps are within this many seconds,
                      merge them into the same window.
        min_duration: Windows shorter than this (in seconds) are discarded.

    Returns:
        List of {"start_ts", "end_ts", "duration_s", "avg_confidence"} dicts.
    """
    if not flag_points:
        return []

    flag_points = sorted(flag_points, key=lambda x: x[0])
    groups: List[List[Tuple[float, float]]] = [[flag_points[0]]]

    for pt in flag_points[1:]:
        if pt[0] - groups[-1][-1][0] <= merge_gap:
            groups[-1].append(pt)
        else:
            groups.append([pt])

    result = []
    for g in groups:
        start = g[0][0]
        end = g[-1][0]
        dur = end - start
        if dur < min_duration:
            continue
        avg_conf = sum(c for _, c in g) / len(g)
        result.append({
            "start_ts": round(start, 3),
            "end_ts": round(end, 3),
            "duration_s": round(dur, 3),
            "avg_confidence": round(avg_conf, 4),
        })
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_report(
    *,
    job_id: str,
    video_path: str,
    candidate_id: str,
    frame_detections: List[Dict[str, Any]],
    audio_speech_segments: List[Tuple[float, float]],
    voice_result: Optional[Dict[str, Any]],
    has_audio: bool,
    video_duration_s: Optional[float] = None,
    frame_count: int = 0,
) -> Dict[str, Any]:
    """
    Build the full JSON report dictionary from all detection outputs.

    Parameters
    ----------
    job_id                : Unique job identifier.
    video_path            : Local path (or URI) of the source video.
    candidate_id          : Candidate / session identifier.
    frame_detections      : Output of the visual detection loop — one entry per sampled frame:
                              {
                                "timestamp": float,
                                "faces":    [{"box": [...], "confidence": float}, ...],
                                "gaze":     {"yaw": float, "pitch": float, "roll": float},
                                "gadgets":  [{"box": [...], "class_name": str, "confidence": float}, ...],
                                "identity_mismatch": float   (optional, 0-1 mismatch confidence)
                              }
    audio_speech_segments : [(start_s, end_s), ...] from VAD.
    voice_result          : Output of count_distinct_voices() or None if no audio.
    has_audio             : Whether audio extraction succeeded.
    video_duration_s      : Total video duration in seconds (if known).
    frame_count           : Total number of frames sampled.

    Returns
    -------
    A dict ready to be serialised to JSON.
    """

    # ----------------------------------------------------------------
    # 1. META
    # ----------------------------------------------------------------
    meta: Dict[str, Any] = {
        "job_id": job_id,
        "candidate_id": candidate_id,
        "video_source": os.path.basename(video_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frames_sampled": frame_count,
        "video_duration_s": round(video_duration_s, 3) if video_duration_s else None,
        "gaze_away_thresholds": {
            "yaw_deg": settings.GAZE_AWAY_YAW_THRESHOLD,
            "pitch_deg": settings.GAZE_AWAY_PITCH_THRESHOLD,
        },
    }

    # ----------------------------------------------------------------
    # 2. FACE TIMELINE — one entry per frame
    # ----------------------------------------------------------------
    face_timeline: List[Dict[str, Any]] = []
    no_face_flags: List[Tuple[float, float]] = []
    multi_face_flags: List[Tuple[float, float]] = []

    for fd in frame_detections:
        ts = fd["timestamp"]
        faces = fd.get("faces", [])
        face_count = len(faces)

        face_timeline.append({
            "timestamp_s": round(ts, 3),
            "face_count": face_count,
            "faces": [
                {
                    "box": f["box"],
                    "confidence": round(float(f["confidence"]), 4),
                }
                for f in faces
            ],
        })

        if face_count == 0:
            no_face_flags.append((ts, 1.0))
        elif face_count >= 2:
            avg_conf = sum(f["confidence"] for f in faces) / face_count
            multi_face_flags.append((ts, avg_conf))

    face_counts = [e["face_count"] for e in face_timeline]
    face_stats: Dict[str, Any] = {
        "frames_with_no_face": sum(1 for c in face_counts if c == 0),
        "frames_with_one_face": sum(1 for c in face_counts if c == 1),
        "frames_with_multiple_faces": sum(1 for c in face_counts if c >= 2),
        "max_faces_in_single_frame": max(face_counts) if face_counts else 0,
    }

    # ----------------------------------------------------------------
    # 3. GAZE TIMELINE — one entry per frame where a face was detected
    # ----------------------------------------------------------------
    gaze_timeline: List[Dict[str, Any]] = []
    gaze_away_flags: List[Tuple[float, float]] = []
    direction_time = {"Left": 0.0, "Right": 0.0, "Straight": 0.0}
    prev_ts = 0.0

    for fd in frame_detections:
        ts = fd["timestamp"]
        dur = ts - prev_ts
        prev_ts = ts
        
        if not fd.get("faces"):
            continue  # no face → gaze not meaningful

        gaze = fd.get("gaze", {"yaw": 0.0, "pitch": 0.0, "roll": 0.0})
        yaw = float(gaze.get("yaw", 0.0))
        pitch = float(gaze.get("pitch", 0.0))
        roll = float(gaze.get("roll", 0.0))
        away = (
            abs(yaw) > settings.GAZE_AWAY_YAW_THRESHOLD
            or abs(pitch) > settings.GAZE_AWAY_PITCH_THRESHOLD
        )

        if yaw > 10.0:
            direction = "Left"
        elif yaw < -10.0:
            direction = "Right"
        else:
            direction = "Straight"
            
        direction_time[direction] += dur

        gaze_timeline.append({
            "timestamp_s": round(ts, 3),
            "yaw_deg": round(yaw, 2),
            "pitch_deg": round(pitch, 2),
            "roll_deg": round(roll, 2),
            "direction": direction,
            "looking_away": away,
        })

        if away:
            gaze_away_flags.append((ts, 0.8))

    gaze_stats: Dict[str, Any] = {
        "frames_with_gaze_data": len(gaze_timeline),
        "frames_looking_away": sum(1 for e in gaze_timeline if e["looking_away"]),
        "time_looking_straight_s": round(direction_time["Straight"], 2),
        "time_looking_left_s": round(direction_time["Left"], 2),
        "time_looking_right_s": round(direction_time["Right"], 2),
    }

    # ----------------------------------------------------------------
    # 4. GADGET TIMELINE — one entry per frame with detections
    # ----------------------------------------------------------------
    gadget_timeline: List[Dict[str, Any]] = []
    gadget_flags: List[Tuple[float, float]] = []

    for fd in frame_detections:
        ts = fd["timestamp"]
        gadgets = fd.get("gadgets", [])
        if not gadgets:
            continue

        gadget_timeline.append({
            "timestamp_s": round(ts, 3),
            "detections": [
                {
                    "class_name": g["class_name"],
                    "confidence": round(float(g["confidence"]), 4),
                    "box": g["box"],
                }
                for g in gadgets
            ],
        })
        best_conf = max(float(g["confidence"]) for g in gadgets)
        gadget_flags.append((ts, best_conf))

    gadget_stats: Dict[str, Any] = {
        "frames_with_gadget_detected": len(gadget_timeline),
        "unique_classes_detected": sorted(
            set(
                d["class_name"]
                for e in gadget_timeline
                for d in e["detections"]
            )
        ),
    }

    # ----------------------------------------------------------------
    # 5. IDENTITY TIMELINE — one entry per frame with a mismatch result
    # ----------------------------------------------------------------
    identity_timeline: List[Dict[str, Any]] = []
    identity_mismatch_flags: List[Tuple[float, float]] = []

    for fd in frame_detections:
        ts = fd["timestamp"]
        if "identity_mismatch" not in fd:
            continue
        mismatch_conf = float(fd["identity_mismatch"])
        flagged = mismatch_conf >= 0.50  # matches worker threshold (1 - sim < 0.50)

        identity_timeline.append({
            "timestamp_s": round(ts, 3),
            "mismatch_confidence": round(mismatch_conf, 4),
            "flagged": flagged,
        })

        if flagged:
            identity_mismatch_flags.append((ts, mismatch_conf))

    identity_checked = len(identity_timeline) > 0
    identity_stats: Dict[str, Any] = {
        "enrollment_photo_provided": identity_checked,
        "frames_checked": len(identity_timeline),
        "frames_flagged_as_mismatch": sum(1 for e in identity_timeline if e["flagged"]),
    }

    # ----------------------------------------------------------------
    # 6. AUDIO — VAD speech segments + diarization
    # ----------------------------------------------------------------
    audio_section: Dict[str, Any] = {"available": has_audio}

    if has_audio:
        speech_seg_list = [
            {
                "start_ts": round(s, 3),
                "end_ts": round(e, 3),
                "duration_s": round(e - s, 3),
            }
            for s, e in audio_speech_segments
        ]
        total_speech = sum(e - s for s, e in audio_speech_segments)
        audio_section["speech_segments"] = speech_seg_list
        audio_section["total_speech_duration_s"] = round(total_speech, 3)
        audio_section["speech_segment_count"] = len(speech_seg_list)

        if voice_result:
            speaker_segments_serialisable: Dict[str, Any] = {}
            for spk, segs in voice_result.get("speaker_segments", {}).items():
                speaker_segments_serialisable[spk] = [
                    {"start_ts": round(s, 3), "end_ts": round(e, 3)}
                    for s, e in segs
                ]

            flagged_segs = [
                {"start_ts": round(s, 3), "end_ts": round(e, 3)}
                for s, e in voice_result.get("flagged_segments", [])
            ]

            audio_section["diarization"] = {
                "num_distinct_speakers": voice_result.get("num_speakers", 1),
                "clustering_confidence": round(float(voice_result.get("confidence", 0.0)), 4),
                "speaker_segments": speaker_segments_serialisable,
                "non_primary_speaker_segments": flagged_segs,
                "non_primary_segment_count": len(flagged_segs),
            }
    else:
        audio_section["note"] = (
            "No audio track found in video — audio analysis skipped."
        )

    # ----------------------------------------------------------------
    # 7. EVENTS — consolidated time windows for each signal type
    # ----------------------------------------------------------------
    second_voice_flags: List[Tuple[float, float]] = []
    if has_audio and voice_result:
        diar_conf = float(voice_result.get("confidence", 0.8))
        second_voice_flags = [
            (s, diar_conf)
            for s, _ in voice_result.get("flagged_segments", [])
        ]

    events: Dict[str, Any] = {
        "no_face_detected": {
            "description": "Periods where no face was visible in the frame.",
            "windows": _ts_windows(no_face_flags, merge_gap=5.0, min_duration=2.0),
        },
        "multiple_faces_detected": {
            "description": "Periods where more than one face was visible simultaneously.",
            "windows": _ts_windows(multi_face_flags, merge_gap=5.0, min_duration=1.5),
        },
        "gaze_away_sustained": {
            "description": (
                f"Periods where head pose exceeded thresholds "
                f"(|yaw| > {settings.GAZE_AWAY_YAW_THRESHOLD}° or "
                f"|pitch| > {settings.GAZE_AWAY_PITCH_THRESHOLD}°)."
            ),
            "windows": _ts_windows(gaze_away_flags, merge_gap=4.0, min_duration=3.0),
        },
        "prohibited_device_detected": {
            "description": "Periods where a prohibited gadget / object was detected.",
            "windows": _ts_windows(gadget_flags, merge_gap=6.0, min_duration=0.0),
        },
        "identity_mismatch": {
            "description": (
                "Periods where the detected face did not match the enrollment photo "
                "(mismatch_confidence >= 0.50, i.e. ArcFace similarity < 0.50)."
            ),
            "windows": _ts_windows(
                identity_mismatch_flags, merge_gap=5.0, min_duration=0.0
            ),
        },
        "second_voice_detected": {
            "description": (
                "Audio segments attributed to a non-primary speaker "
                "(potential additional person in the room)."
            ),
            "windows": _ts_windows(second_voice_flags, merge_gap=5.0, min_duration=1.5),
        },
    }

    # ----------------------------------------------------------------
    # 8. Summary audio sub-section
    # ----------------------------------------------------------------
    audio_summary: Dict[str, Any] = {
        "available": has_audio,
        "total_speech_s": audio_section.get("total_speech_duration_s"),
        "distinct_speakers": (
            audio_section.get("diarization", {}).get("num_distinct_speakers")
            if has_audio else None
        ),
        "non_primary_segments": (
            audio_section.get("diarization", {}).get("non_primary_segment_count")
            if has_audio else None
        ),
    }

    # ----------------------------------------------------------------
    # 9. Assemble final report
    # ----------------------------------------------------------------
    return {
        "meta": meta,
        "summary": {
            "face": face_stats,
            "gaze": gaze_stats,
            "gadget": gadget_stats,
            "identity": identity_stats,
            "audio": audio_summary,
        },
        "events": events,
        "face_timeline": face_timeline,
        "gaze_timeline": gaze_timeline,
        "gadget_timeline": gadget_timeline,
        "identity_timeline": identity_timeline,
        "audio": audio_section,
    }


def save_report(report: Dict[str, Any], output_path: str) -> str:
    """
    Serialises the report dict to a pretty-printed JSON file.

    Returns the absolute path of the written file.
    """
    abs_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return abs_path
