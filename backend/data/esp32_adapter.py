"""
ESP32 Hardware Connector Stub.
Implements the SensorSource interface for future physical ESP32 prototype integration
(with MAX30102 PPG pulse oximeter & MPU6050 6-axis IMU).
"""

import time
import json
from typing import Optional, Dict, Any
import numpy as np

from backend.data.sensor_source import SensorSource, NormalizedPacket
from backend.config import SENSOR_SAMPLING_RATE_HZ, WINDOW_SAMPLES


class ESP32SensorSource(SensorSource):
    """
    Hardware Sensor Source for ESP32 Wearable prototype.
    Receives JSON packets over Serial UART (COM port) or WiFi UDP/WebSocket:
    {
      "timestamp": 1725250000.12,
      "heart_rate": 78.4,
      "eda": 3.2,
      "temperature": 34.1,
      "respiration": 16.0,
      "acceleration": {"x": 0.02, "y": 0.99, "z": 0.08}
    }
    """

    def __init__(self, port: str = "COM3", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self._is_connected = False
        self._is_running = False
        self._buffer = []
        self._latest_packet: Optional[NormalizedPacket] = None

    def connect(self) -> bool:
        """Attempt connection to serial device."""
        try:
            # When pyserial is active on actual hardware:
            # self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"[ESP32SensorSource] Ready to connect to {self.port} at {self.baudrate} baud.")
            self._is_connected = False  # Hardware not connected in demonstration
            return False
        except Exception as e:
            print(f"[ESP32SensorSource] Connection failed: {e}")
            return False

    def start(self) -> None:
        self._is_running = True

    def stop(self) -> None:
        self._is_running = False

    def reset(self) -> None:
        self._buffer.clear()

    def is_running(self) -> bool:
        return self._is_running

    def parse_incoming_json(self, raw_str: str) -> Optional[NormalizedPacket]:
        """Parse incoming JSON payload from ESP32."""
        try:
            data = json.loads(raw_str)
            acc = data.get("acceleration", {})
            ax = float(acc.get("x", 0.0))
            ay = float(acc.get("y", 1.0))
            az = float(acc.get("z", 0.0))
            mag = float(np.sqrt(ax**2 + ay**2 + az**2))

            packet = NormalizedPacket(
                timestamp=float(data.get("timestamp", time.time())),
                subject_id="ESP32_DEV",
                eda=float(data.get("eda", 3.0)),
                ecg=0.0,  # ESP32 MAX30102 provides PPG/HR instead of direct ECG
                respiration=float(data.get("respiration", 16.0)),
                temperature=float(data.get("temperature", 34.0)),
                acc_x=ax,
                acc_y=ay,
                acc_z=az,
                acc_mag=mag,
                heart_rate=float(data.get("heart_rate", 75.0)),
                ground_truth_label=-1,
                source_type="ESP32_LIVE",
            )
            self._latest_packet = packet
            self._buffer.append(packet)
            if len(self._buffer) > WINDOW_SAMPLES * 2:
                self._buffer.pop(0)
            return packet
        except Exception as e:
            print(f"[ESP32] Packet parse error: {e}")
            return None

    def get_next_sample(self) -> Optional[NormalizedPacket]:
        return self._latest_packet

    def get_current_window(self) -> Optional[np.ndarray]:
        if len(self._buffer) < WINDOW_SAMPLES:
            return None
        recent = self._buffer[-WINDOW_SAMPLES:]
        arr = np.array(
            [[p.eda, p.ecg, p.respiration, p.temperature, p.acc_mag] for p in recent],
            dtype=np.float32,
        )
        return arr

    def get_current_state(self) -> Dict[str, Any]:
        return {
            "running": self._is_running,
            "connected": self._is_connected,
            "port": self.port,
            "source_type": "ESP32_LIVE",
            "status_text": "ESP32 Hardware Disconnected (Using WESAD Replay)" if not self._is_connected else "ESP32 Streaming",
        }
