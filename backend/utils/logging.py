"""
Safety Event Logger & Auditor.
Maintains rolling timestamped event log with JSON and CSV export capabilities.
"""

import time
import csv
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from backend.config import LOGS_DIR


class SystemEventLogger:
    """
    Centralized event logger for safety alerts, state transitions, and system telemetry.
    """

    def __init__(self, max_entries: int = 200):
        self.max_entries = max_entries
        self.logs: List[Dict[str, Any]] = []
        self._init_session()

    def _init_session(self) -> None:
        self.log_event("SYSTEM", "Personal Safety Monitor initialized.")
        self.log_event("WEARABLE", "WESAD Sensor Replayer connected.")
        self.log_event("VISION", "Computer Vision subsystem online.")

    def log_event(self, category: str, message: str, level: str = "INFO", meta: Dict[str, Any] = None) -> Dict[str, Any]:
        """Record an event with formatted timestamp."""
        now = datetime.now()
        entry = {
            "id": len(self.logs) + 1,
            "timestamp_str": now.strftime("%H:%M:%S"),
            "timestamp_iso": now.isoformat(),
            "category": category.upper(),
            "level": level.upper(),
            "message": message,
            "meta": meta or {},
        }
        self.logs.append(entry)
        if len(self.logs) > self.max_entries:
            self.logs.pop(0)

        print(f"[{entry['timestamp_str']}] [{entry['category']}] {entry['message']}")
        return entry

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.logs[-limit:]

    def export_json(self) -> Path:
        """Export logs to JSON file."""
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = LOGS_DIR / f"event_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(file_path, "w") as f:
            json.dump(self.logs, f, indent=2)
        return file_path

    def export_csv(self) -> Path:
        """Export logs to CSV file."""
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = LOGS_DIR / f"event_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "timestamp_str", "category", "level", "message"])
            writer.writeheader()
            for row in self.logs:
                writer.writerow({
                    "id": row["id"],
                    "timestamp_str": row["timestamp_str"],
                    "category": row["category"],
                    "level": row["level"],
                    "message": row["message"],
                })
        return file_path


# Global singleton instance
event_logger = SystemEventLogger()
