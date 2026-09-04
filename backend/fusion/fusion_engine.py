"""
Multimodal Fusion Engine.
Fuses Sensor Distress Score and Visual Distress Score with strict false-alarm suppression
and calibrated multimodal agreement escalation.
"""

from typing import Dict, Any, List
import numpy as np

from backend.config import (
    SENSOR_WEIGHT,
    VISION_WEIGHT,
    SAFE_UPPER_BOUND,
    CAUTION_UPPER_BOUND,
    THREAT_LOWER_BOUND,
    ROLLING_WINDOW_SIZE,
)


class FusionEngine:
    """
    Multimodal Decision & Fusion Engine.
    Implements weighted fusion, minimum-agreement enhancement,
    single-modality false-alarm suppression, and temporal smoothing.
    """

    def __init__(
        self,
        sensor_weight: float = SENSOR_WEIGHT,
        vision_weight: float = VISION_WEIGHT,
        rolling_window_size: int = ROLLING_WINDOW_SIZE,
    ):
        self.sensor_weight = sensor_weight
        self.vision_weight = vision_weight
        self.rolling_window_size = rolling_window_size
        self._score_history: List[float] = []

    def set_weights(self, sensor_w: float, vision_w: float) -> None:
        total = sensor_w + vision_w
        if total > 0:
            self.sensor_weight = sensor_w / total
            self.vision_weight = vision_w / total

    def fuse(
        self,
        sensor_distress: float,
        vision_distress: float,
        demo_override: bool = False,
        demo_threat_level: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Calculates raw fused score, agreement factor, smoothed score, and verdict category.
        """
        s_score = float(np.clip(sensor_distress, 0.0, 1.0))
        v_score = float(np.clip(vision_distress, 0.0, 1.0))

        if demo_override:
            raw_fused = float(np.clip(demo_threat_level, 0.0, 1.0))
            agreement_status = "DEMO_MODE_FORCED"
        else:
            # Base Weighted Linear Fusion
            raw_fused = (self.sensor_weight * s_score) + (self.vision_weight * v_score)

            # Multimodal Agreement & False-Positive Suppression Logic
            if s_score >= 0.65 and v_score >= 0.60:
                # High Sensor + High Vision -> Strong Mutual Escalation (THREAT)
                raw_fused = min(1.0, max(0.78, raw_fused * 1.12))
                agreement_status = "MUTUAL_AGREEMENT_HIGH"

            elif s_score >= 0.60 and v_score <= 0.30:
                # High Sensor + Calm Vision -> Physiological Anomaly / Exercise / False Alarm
                # Suppress score to CAUTION ceiling (0.50)
                raw_fused = min(raw_fused, 0.50)
                agreement_status = "PHYSIOLOGICAL_ANOMALY (NON_THREAT)"

            elif v_score >= 0.60 and s_score <= 0.30:
                # High Vision + Calm Sensor -> Benign Visual Anomaly / Stretch / Reaching
                # Suppress score to CAUTION ceiling (0.46)
                raw_fused = min(raw_fused, 0.46)
                agreement_status = "VISUAL_ANOMALY (BENIGN_MOTION)"

            else:
                agreement_status = "BASELINE_CONGRUENT"

        # Temporal Rolling Smoothing
        self._score_history.append(raw_fused)
        if len(self._score_history) > self.rolling_window_size:
            self._score_history.pop(0)

        smoothed_score = float(np.mean(self._score_history))

        # Categorical Verdict
        if smoothed_score <= SAFE_UPPER_BOUND:
            verdict = "SAFE"
            verdict_color = "#ccff00"  # Neon Green
        elif smoothed_score <= CAUTION_UPPER_BOUND:
            verdict = "SUSPICIOUS / CAUTION"
            verdict_color = "#00f0ff"  # Cyan
        else:
            verdict = "THREAT / EMERGENCY"
            verdict_color = "#ff003c"  # Alert Red

        return {
            "sensor_score": round(s_score, 3),
            "vision_score": round(v_score, 3),
            "sensor_weight": round(self.sensor_weight, 2),
            "vision_weight": round(self.vision_weight, 2),
            "raw_fused_score": round(raw_fused, 3),
            "threat_score": round(smoothed_score, 3),
            "threat_score_pct": round(smoothed_score * 100, 1),
            "agreement_status": agreement_status,
            "verdict": verdict,
            "verdict_color": verdict_color,
        }
