"""Verify the BLE wearable adapter's JSON parsing -> NormalizedPacket without hardware."""
import time
from backend.data.ble_adapter import BLEWearableSensorSource

src = BLEWearableSensorSource(device_name="ESA-WEAR-01")
assert src.is_connected is False
assert src.get_next_sample() is None

# Feed the exact JSON schema emitted by the ESP32-C3 firmware.
raw = (
    '{"hr":78,"spo2":98.0,"ax":0.02,"ay":0.99,"az":0.10,'
    '"gx":1.1,"gy":-0.4,"gz":0.8,"mag":1.00,"finger":1}'
)
p = src.parse_incoming_json(raw)
src.inject_packet(p)
assert p is not None, "parser returned None for valid JSON"

# Core fields the fusion/UI rely on.
assert p.heart_rate == 78.0
assert p.spo2 == 98.0
assert p.finger_detected is True
assert abs(p.acc_mag - 1.00) < 1e-6
assert p.source_type == "ARDUINO_LIVE"

# The window-based hardware inference expects a list of packets.
assert isinstance(src.get_current_window(), list)
assert src.get_current_window()[0].heart_rate == 78.0

state = src.get_current_state()
# "connected" reflects a live BLE link; with no hardware we only verify packet capture.
assert state["connected"] is False
assert state["packets_received"] == 1
assert state["is_stale"] is False
assert state["source_type"] == "ARDUINO_LIVE"

# Graceful handling of garbage must not crash.
assert src.parse_incoming_json("not json") is None
assert src.parse_incoming_json('{"hr":"bad"}') is None

print("BLE_ADAPTER_PARSE_OK")
print("  hr=", p.heart_rate, "spo2=", p.spo2, "acc_mag=", p.acc_mag,
      "gyro_mag=", p.gyro_mag, "finger=", p.finger_detected)