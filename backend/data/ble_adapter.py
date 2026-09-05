"""
BLE Wearable Sensor Source (ESP32-C3 Tier 1).

Replaces the tethered USB-serial Arduino source with a wireless Bluetooth
Low Energy link. Reads the same telemetry JSON this project already parses
(``{"hr":..,"spo2":..,"ax":..,...,"mag":..,"finger":..}``) but streams it over
the ``ESA-WEAR-01`` BLE Notify characteristic instead of a serial port.

Implements the ``SensorSource`` ABC so it can be swapped into the fusion/UI
pipeline unchanged (see ``firmware/esp32_wearable/esp32_wearable.ino``).

Dependency: ``pip install bleak`` (added to requirements.txt).
"""

import asyncio
import json
import threading
import time
from typing import Optional, Dict, Any, List

import numpy as np

from backend.data.sensor_source import SensorSource, NormalizedPacket

# ---- Default BLE addressing (match the firmware) --------------------------
DEFAULT_DEVICE_NAME = "ESA-WEAR-01"

try:
    import bleak  # noqa: F401
    BLE_AVAILABLE = True
except ImportError:
    BLE_AVAILABLE = False


class BLEWearableSensorSource(SensorSource):
    """
    Wireless wearable source reading JSON telemetry over BLE Notify.
    Runs its own asyncio loop in a background thread (mirror of the
    Arduino serial reader pattern) to stay non-blocking.
    """

    def __init__(
        self,
        device_name: str = DEFAULT_DEVICE_NAME,
        char_uuid: str = "8d54b3a0-0001-0000-0000-ffff0000c3ec",
    ):
        self.device_name = device_name
        self.char_uuid = char_uuid.lower()

        self._is_connected = False
        self._is_running = True
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._latest_packet: Optional[NormalizedPacket] = None
        self._buffer: List[NormalizedPacket] = []
        self._packets_received = 0
        self._last_packet_time = 0.0
        self._connection_error = ""

    # ---- SensorSource interface -------------------------------------------
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

    # ---- Connection lifecycle ---------------------------------------------
    def connect(self, timeout_s: float = 15.0) -> bool:
        if not BLE_AVAILABLE:
            self._connection_error = "bleak library not installed (pip install bleak)"
            return False

        with self._lock:
            if self._is_connected and self._thread and self._thread.is_alive():
                return True
            self._is_connected = True
            self._connection_error = ""
            self._thread = threading.Thread(
                target=self._run_ble_loop, args=(timeout_s,), daemon=True
            )
            self._thread.start()
            return True

    def disconnect(self) -> None:
        with self._lock:
            self._is_connected = False
        if self._loop and self._loop.is_running():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
# ---- Packet parsing (identical semantics to the Arduino adapter) --------
    def parse_incoming_json(self, raw_str: str) -> Optional[NormalizedPacket]:
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

            acc_mag = float(data.get("mag", np.sqrt(ax ** 2 + ay ** 2 + az ** 2)))
            gyro_mag = float(np.sqrt(gx ** 2 + gy ** 2 + gz ** 2))
            finger = bool(int(data.get("finger", 1)))

            packet = NormalizedPacket(
                timestamp=time.time(),
                subject_id="ARDUINO_LIVE",  # keeps UI hardware panel keys working
                eda=0.0,
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
        with self._lock:
            self._latest_packet = packet
            self._last_packet_time = time.time()
            self._packets_received += 1
            self._buffer.append(packet)
            if len(self._buffer) > 120:
                self._buffer.pop(0)

    # ---- Data accessors -----------------------------------------------------
    def get_next_sample(self) -> Optional[NormalizedPacket]:
        with self._lock:
            return self._latest_packet

    def get_current_window(self) -> Optional[List[NormalizedPacket]]:
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
                "device": self.device_name,
                "char_uuid": self.char_uuid,
                "packets_received": self._packets_received,
                "last_packet_age_s": round(now - self._last_packet_time, 2) if self._last_packet_time > 0 else None,
                "is_stale": is_stale,
                "error": self._connection_error,
                "source_type": "ARDUINO_LIVE",
            }
# ---- Background BLE loop ------------------------------------------------
    def _store_packet(self, byte_data: Optional[bytearray]) -> None:
        if byte_data is None:
            return
        try:
            raw = byte_data.decode("utf-8", errors="ignore")
        except Exception:
            return
        packet = self.parse_incoming_json(raw)
        if packet:
            with self._lock:
                self._latest_packet = packet
                self._last_packet_time = time.time()
                self._packets_received += 1
                self._buffer.append(packet)
                if len(self._buffer) > 120:
                    self._buffer.pop(0)

    def _run_ble_loop(self, timeout_s: float) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._ble_worker(timeout_s))
        except Exception as e:
            self._connection_error = str(e)
            print(f"[BLE] Connection thread ended: {e}")
        finally:
            with self._lock:
                self._is_connected = False
            self._loop.close()

    async def _find_device(self):
        from bleak import BleakScanner

        async def scan():
            devices = await BleakScanner.discover(timeout=10.0)
            for d in devices:
                if d.name and d.name.lower() == self.device_name.lower():
                    return d
                adv = getattr(d, "details", None) and d.details.get("adv")
                if adv and getattr(adv, "local_name", "").lower() == self.device_name.lower():
                    return d
            return None

        return await scan()

    async def _ble_worker(self, timeout_s: float) -> None:
        from bleak import BleakClient

        device = await self._find_device()
        if device is None:
            self._connection_error = f"Device '{self.device_name}' not found (is it on?)"
            print(f"[BLE] {self._connection_error}")
            raise RuntimeError(self._connection_error)

        print(f"[BLE] Connecting to {device.name} @ {device.address} ...")
        async with BleakClient(device.address, timeout=timeout_s) as client:
            print(f"[BLE] Connected: {client.is_connected}")
            with self._lock:
                self._connection_error = ""

            def _on_notify(_characteristic, data):
                self._store_packet(data)

            await client.start_notify(self.char_uuid, _on_notify)
            print(f"[BLE] Subscribed to Notify on {self.char_uuid}. Streaming live data.")
            while self._is_connected and self._is_running and client.is_connected:
                await asyncio.sleep(0.1)
            try:
                await client.stop_notify(self.char_uuid)
            except Exception:
                pass
        print(f"[BLE] Disconnected from {self.device_name}.")