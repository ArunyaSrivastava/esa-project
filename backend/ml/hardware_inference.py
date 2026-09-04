"""
Dedicated Hardware Distress Inference Engine.
Evaluates genuine physical sensor signals from Arduino Uno (MAX30102 PPG/SpO2 + MPU6050 6-axis IMU)
without fabricated EDA, ECG, respiration, or skin temperature.
"""

import time
from typing import Dict, Any, Optional, List
import numpy as np

from backend.data.sensor_source import NormalizedPacket


class HardwareDistressInference:
    """
    Real-time inference manager for physical Arduino wearable hardware.
    Analyzes true heart rate dynamics, SpO2 hypoxia, impact shocks, and violent struggle.
    """

    def __init__(self):
        self._hr_history: List[float] = []
        self._acc_history: List[float] = []
        self._gyro_history: List[float] = []
        self._last_impact_time: float = 0.0

    def predict(self, packet: Optional[NormalizedPacket], window: Optional[List[NormalizedPacket]] = None) -> Dict[str, Any]:
        """
        Takes the latest NormalizedPacket from Arduino and returns calibrated hardware distress metrics.
        """
        t0 = time.perf_counter()

        if packet is None:
            return {
                "distress_score": 0.05,
                "distress_pct": 5.0,
                "probabilities": {"baseline": 0.90, "tachycardia": 0.06, "motion_distress": 0.04},
                "predicted_label": "No Data",
                "hr_bpm": 0.0,
                "spo2_pct": 0.0,
                "motion_mag_g": 0.0,
                "gyro_dps": 0.0,
                "finger_detected": False,
                "latency_ms": 0.1,
                "sub_metrics": {
                    "physiological_distress": 0.05,
                    "motion_distress": 0.05,
                    "impact_shock": False,
                    "struggle_detected": False,
                }
            }

        hr = packet.heart_rate or 0.0
        spo2 = packet.spo2 or 0.0
        acc_mag = packet.acc_mag
        gyro_mag = packet.gyro_mag
        finger = packet.finger_detected

        # Update sliding history
        if hr > 30.0:
            self._hr_history.append(hr)
            if len(self._hr_history) > 30:
                self._hr_history.pop(0)

        self._acc_history.append(acc_mag)
        if len(self._acc_history) > 30:
            self._acc_history.pop(0)

        self._gyro_history.append(gyro_mag)
        if len(self._gyro_history) > 30:
            self._gyro_history.pop(0)

        # 1. Physiological Distress Evaluation (MAX30102)
        physio_distress = 0.05
        tachycardia_active = False

        if finger and hr >= 40.0:
            # HR Score curve
            if hr > 125.0:
                physio_distress = 0.88
                tachycardia_active = True
            elif hr > 105.0:
                physio_distress = 0.65
                tachycardia_active = True
            elif hr > 90.0:
                physio_distress = 0.35
            elif hr > 80.0:
                physio_distress = 0.15
            else:
                physio_distress = 0.05

            # SpO2 penalty
            if 0 < spo2 < 90.0:
                physio_distress = min(1.0, physio_distress + 0.35)
            elif 0 < spo2 < 95.0:
                physio_distress = min(1.0, physio_distress + 0.15)
        else:
            # Finger not placed on sensor -> safe baseline, do not false alarm
            physio_distress = 0.05

        # 2. Biomechanical Motion & Impact Shock Evaluation (MPU6050)
        motion_distress = 0.05
        impact_shock = False
        struggle_active = False

        # Impact spike detection (|a| > 2.6g or sudden shock)
        if acc_mag > 2.6 or acc_mag < 0.20:
            self._last_impact_time = time.time()
            impact_shock = True

        # Recent impact within last 3 seconds
        if (time.time() - self._last_impact_time) < 3.0:
            impact_shock = True
            motion_distress = max(motion_distress, 0.75)

        # Violent struggle rotation detection (gyro > 140 deg/s)
        avg_gyro = float(np.mean(self._gyro_history)) if len(self._gyro_history) > 0 else 0.0
        if avg_gyro > 130.0 or gyro_mag > 180.0:
            struggle_active = True
            motion_distress = max(motion_distress, 0.60)
        elif avg_gyro > 60.0:
            motion_distress = max(motion_distress, 0.25)

        # 3. Multimodal Distress Score Synthesis
        # If either high tachycardia or severe physical shock occurs, reflect in sensor distress
        combined_score = max(physio_distress, motion_distress, (0.60 * physio_distress + 0.40 * motion_distress))
        combined_score = float(np.clip(combined_score, 0.02, 1.0))

        # 4. Probabilities derivation
        if tachycardia_active:
            p_tach = float(np.clip(physio_distress, 0.40, 0.90))
            p_mot = float(np.clip(motion_distress, 0.05, 0.30))
            p_base = max(0.01, 1.0 - (p_tach + p_mot))
            pred_label = "Tachycardia / Elevated HR"
        elif impact_shock:
            p_mot = float(np.clip(motion_distress, 0.60, 0.92))
            p_tach = float(np.clip(physio_distress, 0.05, 0.20))
            p_base = max(0.01, 1.0 - (p_tach + p_mot))
            pred_label = "Impact Shock / Fall"
        elif struggle_active:
            p_mot = float(np.clip(motion_distress, 0.50, 0.85))
            p_tach = float(np.clip(physio_distress, 0.05, 0.25))
            p_base = max(0.01, 1.0 - (p_tach + p_mot))
            pred_label = "Violent Agitation / Struggle"
        else:
            p_base = float(np.clip(1.0 - combined_score, 0.70, 0.98))
            p_tach = float(np.clip(physio_distress, 0.01, 0.15))
            p_mot = float(np.clip(motion_distress, 0.01, 0.15))
            pred_label = "Normal Baseline"

        # Re-normalize probabilities
        tot = p_base + p_tach + p_mot
        p_base /= tot
        p_tach /= tot
        p_mot /= tot

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "distress_score": round(combined_score, 3),
            "distress_pct": round(combined_score * 100, 1),
            "probabilities": {
                "baseline": round(p_base, 3),
                "stress": round(p_tach, 3),  # mapped to stress bar in UI
                "amusement": round(p_mot, 3),  # mapped to motion/amusement bar in UI
            },
            "predicted_label": pred_label,
            "hr_bpm": hr,
            "spo2_pct": spo2,
            "motion_mag_g": acc_mag,
            "gyro_dps": gyro_mag,
            "finger_detected": finger,
            "latency_ms": latency_ms,
            "sub_metrics": {
                "physiological_distress": round(physio_distress, 3),
                "motion_distress": round(motion_distress, 3),
                "impact_shock": impact_shock,
                "struggle_detected": struggle_active,
            }
        }
