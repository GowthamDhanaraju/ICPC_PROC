import math
from typing import List, Dict, Any

class ScoringEngine:
    def __init__(self):
        # Base severity weights for violation types
        self.base_penalties = {
            "IDENTITY_MISMATCH": 40.0,
            "MULTIPLE_FACES": 25.0,
            "PROHIBITED_DEVICE": 25.0,
            "SECOND_VOICE_DETECTED": 25.0,
            "GAZE_AWAY_SUSTAINED": 10.0,
            "NO_FACE_DETECTED": 10.0
        }
        
        # Identify whether a violation's penalty scales with duration (True) or occurrence count (False)
        self.is_duration_based = {
            "IDENTITY_MISMATCH": False,
            "MULTIPLE_FACES": False,       # Count-based count of entries
            "PROHIBITED_DEVICE": False,     # Occurrence count
            "SECOND_VOICE_DETECTED": True, # Scales with duration
            "GAZE_AWAY_SUSTAINED": True,   # Scales with duration
            "NO_FACE_DETECTED": True       # Scales with duration
        }

    def calculate_score(self, violations: List[Dict[str, Any]]) -> float:
        """
        Calculates a fairness score from 0 to 100.
        Starts at 100 and subtracts penalties using diminishing marginal severity rules.
        """
        score = 100.0

        # Group violations by type
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for v in violations:
            grouped.setdefault(v["type"], []).append(v)

        total_penalty = 0.0

        for vtype, events in grouped.items():
            base_p = self.base_penalties.get(vtype, 10.0)
            
            if self.is_duration_based.get(vtype, False):
                # Duration based penalty: base_p * sqrt(total_duration / 5.0)
                # This yields a sub-linear (diminishing) marginal penalty for long durations
                total_duration = sum(e["duration"] for e in events)
                if total_duration > 0:
                    penalty = base_p * math.sqrt(total_duration / 5.0)
                    total_penalty += penalty
            else:
                # Count based penalty with geometric decay:
                # 1st: base_p
                # 2nd: base_p * 0.5
                # 3rd: base_p * 0.25
                # ...
                decay = 0.5
                v_penalty = 0.0
                for i in range(len(events)):
                    v_penalty += base_p * (decay ** i)
                total_penalty += v_penalty

        score -= total_penalty
        
        # Clamp score between 0.0 and 100.0
        return max(0.0, min(100.0, round(score, 2)))
