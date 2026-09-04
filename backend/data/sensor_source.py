"""
SensorSource Abstract Interface and Normalized Packet Schema.
Enables plug-and-play interchangeability between WESAD Replay and future physical ESP32.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class NormalizedPacket:
    timestamp: float
    subject_id: str
    eda: float  # Electrodermal Activity (microSiemens / standardized)
    ecg: float  # Electrocardiogram (mV / standardized)
    respiration: float  # Respiration amplitude (standardized)
    temperature: float  # Skin temperature (Celsius / standardized)
    acc_x: float  # Accelerometer X (g)
    acc_y: float  # Accelerometer Y (g)
    acc_z: float  # Accelerometer Z (g)
    acc_mag: float  # sqrt(x^2 + y^2 + z^2)
    heart_rate: Optional[float] = None  # Estimated heart rate in BPM
    ground_truth_label: int = -1  # 0: Baseline, 1: Stress, 2: Amusement, -1: None
    source_type: str = "WESAD_REPLAY"  # WESAD_REPLAY, SYNTHETIC, ESP32_LIVE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SensorSource(ABC):
    """
    Abstract Base Class for Sensor Sources.
    Both WESAD Replay and future ESP32 drivers implement this contract.
    """

    @abstractmethod
    def start(self) -> None:
        """Start or resume data streaming."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop or pause data streaming."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset stream to beginning."""
        pass

    @abstractmethod
    def get_next_sample(self) -> Optional[NormalizedPacket]:
        """Fetch the next instantaneous sensor packet."""
        pass

    @abstractmethod
    def get_current_window(self) -> Optional[np.ndarray]:
        """Fetch the current window buffer formatted for ML inference (N_samples, N_features)."""
        pass

    @abstractmethod
    def get_current_state(self) -> Dict[str, Any]:
        """Return runtime status (running, subject, speed, index, etc.)."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Return True if currently active."""
        pass
