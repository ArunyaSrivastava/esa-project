"""
Arduino Uno Physical Hardware Adapter.
Manages USB serial communication, telemetry parsing (MAX30102 PPG/SpO2 + MPU6050 6-axis IMU),
and bidirectional transmission of laptop multimodal fusion status to the SSD1306 OLED display.
"""

import time
import json
import threading
from typing import Optional, Dict, Any, List
import numpy as np

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

from backend.data.sensor_source import SensorSource, NormalizedPacket
from backend.config import (
    DEFAULT_ARDUINO_PORT,
    DEFAULT_ARDUINO_BAUD,
    ARDUINO_SAMPLING_RATE_HZ,
)


class ArduinoSensorSource(SensorSource):
    """
    Physical hardware source for Arduino Uno wearing MAX30102 + MPU6050 + SSD1306.
    Receives JSON packets over USB serial and sends fusion decisions back for OLED rendering.
    """

    def __init__(self, port: str = DEFAULT_ARDUINO_PORT, baudrate: int = DEFAULT_ARDUINO_BAUD):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn: Optional[Any] = None
        
        self._is_connected = False
        self._is_running = True
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self._buffer: List[NormalizedPacket] = []
        self._latest_packet: Optional[NormalizedPacket] = None
        self._packets_received = 0
        self._last_packet_time = 0.0
        self._connection_error = ""

    @staticmethod
    def list_available_ports() -> List[Dict[str, str]]:
        """List all available serial COM ports on the system."""
        if not SERIAL_AVAILABLE:
            return []
        ports = []
        try:
            for p in serial.tools.list_ports.comports():
                ports.append({
                    "port": p.device,
                    "description": p.description,
                    "hwid": p.hwid,
                })
        except Exception as e:
            print(f"[Arduino] Error listing ports: {e}")
        return ports

    def connect(self, port: Optional[str] = None) -> bool:
        """Attempt connection to specified or default COM port."""
        if not SERIAL_AVAILABLE:
            self._connection_error = "pyserial library not available"
            return False

        if port:
            self.port = port

        with self._lock:
            if self._is_connected and self.serial_conn and self.serial_conn.is_open:
                return True

            try:
                print(f"[Arduino] Opening serial connection to {self.port} at {self.baudrate} baud...")
                self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.1)
                time.sleep(1.5)  # Allow Arduino DTR reset to complete
                self._is_connected = True
                self._connection_error = ""
                print(f"[Arduino] Successfully connected to {self.port}.")
            except Exception as e:
                self._is_connected = False
                self._connection_error = str(e)
                print(f"[Arduino] Connection failed to {self.port}: {e}")
                return False

        # Start background reader thread if not active
        if self._thread is None or not self._thread.is_alive():
            self._is_running = True
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()

        return True

    def disconnect(self) -> None:
        """Gracefully close serial connection."""
        with self._lock:
            self._is_connected = False
            if self.serial_conn and self.serial_conn.is_open:
                try:
                    self.serial_conn.close()
                except Exception:
                    pass
            self.serial_conn = None
            print(f"[Arduino] Disconnected from {self.port}.")

    def start(self) -> None:
        self._is_running = True

    def stop(self) -> None:
        self._is_running = False

    def reset(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._latest_packet = None

    def is_running(self) -> bool:
        return self._is_running and self._is_connected

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def send_feedback(self, threat_score_pct: int, state_str: str) -> bool:
        """
        Transmit laptop multimodal decision back to Arduino to display on SSD1306 OLED:
        {"threat": 12, "state": "SAFE"}\n
        """
        if not self._is_connected or not self.serial_conn or not self.serial_conn.is_open:
            return False

        try:
            msg = json.dumps({"threat": int(threat_score_pct), "state": str(state_str)}) + "\n"
            self.serial_conn.write(msg.encode("utf-8"))
            return True
        except Exception as e:
            print(f"[Arduino] Error sending OLED feedback: {e}")
            return False

    def parse_incoming_json(self, raw_str: str) -> Optional[NormalizedPacket]:
        """
        Parse raw JSON line from Arduino:
        {"hr":78.4,"spo2":98.1,"ax":0.02,"ay":0.98,"az":0.12,"gx":1.2,"gy":-0.5,"gz":0.3,"mag":0.99,"finger":1}
        """
        raw_str = raw_str.strip()
        if not raw_str.startswith("{") or not raw_str.endswith("}"):
            return None

        try:
            data = json.loads(raw_str)
            
            hr = float(data.get("hr", 0.0))
            spo2 = float(data.get("spo2", 0.0))
            ax = float(data.get("ax", 0.0))
            ay = float(data.get("ay", 0.0))
            az = float(data.get("az", 0.0))
            gx = float(data.get("gx", 0.0))
            gy = float(data.get("gy", 0.0))
            gz = float(data.get("gz", 0.0))
            
            # Derived magnitudes
            acc_mag = float(data.get("mag", np.sqrt(ax**2 + ay**2 + az**2)))
            gyro_mag = float(np.sqrt(gx**2 + gy**2 + gz**2))
            finger = bool(int(data.get("finger", 1)))

            packet = NormalizedPacket(
                timestamp=time.time(),
                subject_id="ARDUINO_LIVE",
                eda=0.0,  # Zero-fabrication: real hardware does not have EDA/ECG
                ecg=0.0,
                respiration=0.0,
                temperature=0.0,
                acc_x=round(ax, 3),
                acc_y=round(ay, 3),
                acc_z=round(az, 3),
                acc_mag=round(acc_mag, 3),
                gyro_x=round(gx, 2),
                gyro_y=round(gy, 2),
                gyro_z=round(gz, 2),
                gyro_mag=round(gyro_mag, 2),
                heart_rate=round(hr, 1) if hr > 0 else None,
                spo2=round(spo2, 1) if spo2 > 0 else None,
                finger_detected=finger,
                ground_truth_label=-1,
                source_type="ARDUINO_LIVE",
            )
            return packet
        except Exception:
            return None

    def inject_packet(self, packet: NormalizedPacket) -> None:
        """Allow manual / test injection of a normalized packet (used for unit testing)."""
        with self._lock:
            self._latest_packet = packet
            self._last_packet_time = time.time()
            self._packets_received += 1
            self._buffer.append(packet)
            if len(self._buffer) > 120:
                self._buffer.pop(0)

    def _reader_loop(self) -> None:
        """Background thread reading lines from Arduino serial port."""
        while self._is_running:
            if not self._is_connected or not self.serial_conn:
                time.sleep(0.2)
                continue

            try:
                line = self.serial_conn.readline().decode("utf-8", errors="ignore")
                if line:
                    packet = self.parse_incoming_json(line)
                    if packet:
                        with self._lock:
                            self._latest_packet = packet
                            self._last_packet_time = time.time()
                            self._packets_received += 1
                            self._buffer.append(packet)
                            if len(self._buffer) > 120:
                                self._buffer.pop(0)
            except serial.SerialException as e:
                print(f"[Arduino] Serial error / disconnected: {e}")
                with self._lock:
                    self._is_connected = False
                    self._connection_error = str(e)
                time.sleep(0.5)
            except Exception as e:
                time.sleep(0.05)

    def get_next_sample(self) -> Optional[NormalizedPacket]:
        with self._lock:
            return self._latest_packet

    def get_current_window(self) -> Optional[List[NormalizedPacket]]:
        """Return the recent list of NormalizedPackets for hardware distress inference."""
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return list(self._buffer)

    def get_current_state(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            is_stale = (now - self._last_packet_time) > 2.0 if self._last_packet_time > 0 else True
            
            return {
                "running": self._is_running,
                "connected": self._is_connected,
                "port": self.port,
                "baudrate": self.baudrate,
                "packets_received": self._packets_received,
                "last_packet_age_s": round(now - self._last_packet_time, 2) if self._last_packet_time > 0 else None,
                "is_stale": is_stale,
                "error": self._connection_error,
                "source_type": "ARDUINO_LIVE",
            }
