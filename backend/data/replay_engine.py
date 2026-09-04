"""
Deterministic WESAD Replay Engine.
Implements the SensorSource interface, providing real-time streaming,
speed control (0.25x-5x), pause/resume, seek, and scenario jumps.
"""

import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

from backend.config import (
    WESAD_DIR,
    DEFAULT_SUBJECT,
    DEFAULT_REPLAY_SPEED,
    SENSOR_SAMPLING_RATE_HZ,
    WINDOW_SAMPLES,
    LABEL_NAMES,
)
from backend.data.sensor_source import SensorSource, NormalizedPacket
from backend.data.wesad_adapter import WESADAdapter


class ReplayEngine(SensorSource):
    """
    Replay engine that reads WESAD processed signals and streams them
    as normalized IoT packets at configurable speeds.
    """

    def __init__(
        self,
        data_dir: Path = WESAD_DIR,
        subject_id: str = DEFAULT_SUBJECT,
        speed: float = DEFAULT_REPLAY_SPEED,
    ):
        self.data_dir = Path(data_dir)
        self.adapter = WESADAdapter(self.data_dir)
        self.subject_id = subject_id
        self.speed = max(0.1, float(speed))
        
        # State variables
        self._is_running = True
        self._current_index = 0
        self._total_samples = 0
        self._processed_data: Optional[Dict[str, np.ndarray]] = None
        self._lock = threading.Lock()
        
        # Telemetry & Packet counters
        self._packets_sent = 0
        self._start_wall_time = time.time()
        self._last_sample_time = time.time()
        
        # Load initial subject
        self.load_subject(self.subject_id)

    def load_subject(self, subject_id: str) -> bool:
        """Load and cache processed subject signals."""
        with self._lock:
            data = self.adapter.process_subject(subject_id, target_hz=SENSOR_SAMPLING_RATE_HZ)
            if data is None:
                print(f"[ReplayEngine] Could not load subject {subject_id}")
                return False

            self.subject_id = subject_id
            self._processed_data = data
            self._total_samples = len(data["ecg"])
            self._current_index = 0
            self._packets_sent = 0
            print(f"[ReplayEngine] Loaded subject {subject_id} ({self._total_samples} samples, {self._total_samples / SENSOR_SAMPLING_RATE_HZ:.1f}s)")
            return True

    def get_available_subjects(self) -> List[str]:
        """List all available subjects on disk."""
        return self.adapter.find_available_subjects()

    def start(self) -> None:
        """Start or resume streaming."""
        with self._lock:
            self._is_running = True
            self._last_sample_time = time.time()

    def stop(self) -> None:
        """Pause streaming."""
        with self._lock:
            self._is_running = False

    def reset(self) -> None:
        """Seek back to start of record (Baseline)."""
        with self._lock:
            self._current_index = 0
            self._packets_sent = 0

    def seek_fraction(self, fraction: float) -> None:
        """Seek to a relative position (0.0 to 1.0)."""
        with self._lock:
            if self._total_samples > 0:
                fraction = np.clip(fraction, 0.0, 1.0)
                self._current_index = int(fraction * (self._total_samples - 1))

    def seek_to_time(self, seconds: float) -> None:
        """Seek to absolute playback time in seconds."""
        with self._lock:
            sample_idx = int(seconds * SENSOR_SAMPLING_RATE_HZ)
            self._current_index = max(0, min(sample_idx, self._total_samples - 1))

    def seek_to_scenario(self, scenario_type: str) -> bool:
        """
        Jump playback directly to a section matching the requested scenario:
        - 'BASELINE': jumps to first label == 0 segment (Calm baseline)
        - 'STRESS': jumps to first label == 1 segment (Physiological stress)
        - 'AMUSEMENT': jumps to first label == 2 segment
        """
        if self._processed_data is None:
            return False

        labels = self._processed_data["label"]
        target_label = -1
        sc_upper = scenario_type.upper()
        if "BASE" in sc_upper or "NORM" in sc_upper:
            target_label = 0
        elif "STRESS" in sc_upper or "THREAT" in sc_upper or "DISTRESS" in sc_upper:
            target_label = 1
        elif "AMUSE" in sc_upper:
            target_label = 2

        if target_label >= 0:
            indices = np.where(labels == target_label)[0]
            if len(indices) > 0:
                with self._lock:
                    jump_idx = min(indices[0] + SENSOR_SAMPLING_RATE_HZ * 4, indices[-1])
                    self._current_index = int(jump_idx)
                return True
        return False

    def set_speed(self, speed: float) -> None:
        """Adjust playback speed multiplier."""
        with self._lock:
            self.speed = max(0.1, min(10.0, float(speed)))

    def is_running(self) -> bool:
        return self._is_running

    def advance(self, step_samples: int = 1) -> None:
        """Advance playback by given number of samples."""
        with self._lock:
            if self._total_samples > 0 and self._is_running:
                self._current_index = (self._current_index + step_samples) % self._total_samples
                self._packets_sent += step_samples

    def get_next_sample(self) -> Optional[NormalizedPacket]:
        """
        Return the latest sensor packet at the current playback position.
        """
        with self._lock:
            if self._processed_data is None or self._total_samples == 0:
                return None

            idx = self._current_index
            data = self._processed_data

            packet = NormalizedPacket(
                timestamp=time.time(),
                subject_id=self.subject_id,
                eda=float(data["eda"][idx]),
                ecg=float(data["ecg"][idx]),
                respiration=float(data["respiration"][idx]),
                temperature=float(data["temperature"][idx]),
                acc_x=float(data["acc_x"][idx]),
                acc_y=float(data["acc_y"][idx]),
                acc_z=float(data["acc_z"][idx]),
                acc_mag=float(data["acc_mag"][idx]),
                heart_rate=float(data["heart_rate"][idx]) if "heart_rate" in data else 72.0,
                ground_truth_label=int(data["label"][idx]),
                source_type="WESAD_REPLAY",
            )

            return packet

    def get_current_window(self) -> Optional[np.ndarray]:
        """
        Extract the trailing window of WINDOW_SAMPLES (128 samples = 4s at 32Hz)
        with 5 features [eda, ecg, respiration, temperature, acc_mag].
        Shape: (WINDOW_SAMPLES, 5).
        """
        with self._lock:
            if self._processed_data is None or self._total_samples == 0:
                return None

            idx = self._current_index
            data = self._processed_data

            if idx >= WINDOW_SAMPLES:
                slice_idx = slice(idx - WINDOW_SAMPLES, idx)
                window_data = np.stack(
                    [
                        data["eda"][slice_idx],
                        data["ecg"][slice_idx],
                        data["respiration"][slice_idx],
                        data["temperature"][slice_idx],
                        data["acc_mag"][slice_idx],
                    ],
                    axis=-1,
                )
            else:
                part1 = np.stack(
                    [
                        data["eda"][self._total_samples - (WINDOW_SAMPLES - idx) :],
                        data["ecg"][self._total_samples - (WINDOW_SAMPLES - idx) :],
                        data["respiration"][self._total_samples - (WINDOW_SAMPLES - idx) :],
                        data["temperature"][self._total_samples - (WINDOW_SAMPLES - idx) :],
                        data["acc_mag"][self._total_samples - (WINDOW_SAMPLES - idx) :],
                    ],
                    axis=-1,
                )
                part2 = np.stack(
                    [
                        data["eda"][:idx],
                        data["ecg"][:idx],
                        data["respiration"][:idx],
                        data["temperature"][:idx],
                        data["acc_mag"][:idx],
                    ],
                    axis=-1,
                )
                window_data = np.concatenate([part1, part2], axis=0)

            return window_data.astype(np.float32)

    def get_current_state(self) -> Dict[str, Any]:
        """Return runtime state dictionary for UI consumption."""
        with self._lock:
            curr_label = -1
            if self._processed_data is not None and self._total_samples > 0:
                curr_label = int(self._processed_data["label"][self._current_index])

            label_name = LABEL_NAMES.get(curr_label, "Unknown")
            elapsed_secs = self._current_index / SENSOR_SAMPLING_RATE_HZ
            total_secs = self._total_samples / SENSOR_SAMPLING_RATE_HZ if self._total_samples > 0 else 0.0

            return {
                "running": self._is_running,
                "subject_id": self.subject_id,
                "speed": self.speed,
                "current_index": self._current_index,
                "total_samples": self._total_samples,
                "elapsed_seconds": round(elapsed_secs, 1),
                "total_seconds": round(total_secs, 1),
                "progress_fraction": round(self._current_index / max(1, self._total_samples), 4),
                "ground_truth_label": curr_label,
                "ground_truth_name": label_name,
                "packets_sent": self._packets_sent,
                "sampling_rate_hz": SENSOR_SAMPLING_RATE_HZ,
                "source_type": "WESAD_REPLAY",
            }
