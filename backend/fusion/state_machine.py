"""
Alert State Machine with Temporal Persistence & Hysteresis.
Prevents rapid state flickering and orchestrates escalation to ALERT ACTIVE.
"""

import time
from typing import Dict, Any, Optional, List

from backend.config import (
    SAFE_UPPER_BOUND,
    CAUTION_UPPER_BOUND,
    THREAT_LOWER_BOUND,
    THREAT_CONSECUTIVE_SECS,
    RECOVERY_CONSECUTIVE_SECS,
)


class AlertStateMachine:
    """
    Finite State Machine for Safety Escalation:
      SAFE -> CAUTION -> THREAT -> ALERT ACTIVE -> RECOVERY / RESET -> SAFE
    """

    STATES = ["SAFE", "CAUTION", "THREAT", "ALERT ACTIVE", "RECOVERY"]

    def __init__(self):
        self.current_state = "SAFE"
        self._state_entered_time = time.time()
        self._threat_timer_start: Optional[float] = None
        self._recovery_timer_start: Optional[float] = None
        self.total_alerts_triggered = 0
        self.emergency_dispatched = False

    def update(self, fused_threat_score: float) -> Optional[Dict[str, Any]]:
        """
        Evaluate new threat score, manage timers, and handle state transitions.
        Returns a log event dict if a transition occurred, else None.
        """
        now = time.time()
        old_state = self.current_state
        event = None

        if self.current_state == "SAFE":
            if fused_threat_score > THREAT_LOWER_BOUND:
                # Begin threat persistence count
                if self._threat_timer_start is None:
                    self._threat_timer_start = now
                elif (now - self._threat_timer_start) >= THREAT_CONSECUTIVE_SECS:
                    self._transition_to("THREAT", now)
                    event = {
                        "level": "CRITICAL",
                        "message": f"Multimodal threat threshold exceeded for {THREAT_CONSECUTIVE_SECS}s. Escalated to THREAT.",
                    }
                else:
                    self._transition_to("CAUTION", now)
                    event = {
                        "level": "WARNING",
                        "message": "Elevated distress detected across modalities. Entered CAUTION.",
                    }
            elif fused_threat_score > SAFE_UPPER_BOUND:
                self._threat_timer_start = None
                self._transition_to("CAUTION", now)
                event = {
                    "level": "WARNING",
                    "message": "Fused distress entered CAUTION band.",
                }
            else:
                self._threat_timer_start = None

        elif self.current_state == "CAUTION":
            if fused_threat_score >= THREAT_LOWER_BOUND:
                if self._threat_timer_start is None:
                    self._threat_timer_start = now
                elif (now - self._threat_timer_start) >= THREAT_CONSECUTIVE_SECS:
                    self._transition_to("THREAT", now)
                    event = {
                        "level": "CRITICAL",
                        "message": f"High distress persisted for {THREAT_CONSECUTIVE_SECS}s. THREAT STATE ENTERED.",
                    }
            else:
                self._threat_timer_start = None
                if fused_threat_score <= SAFE_UPPER_BOUND:
                    self._transition_to("SAFE", now)
                    event = {
                        "level": "INFO",
                        "message": "Distress returned to normal baseline. System SAFE.",
                    }

        elif self.current_state == "THREAT":
            # Auto-escalate to ALERT ACTIVE after 1.5s in THREAT
            if (now - self._state_entered_time) >= 1.5:
                self._transition_to("ALERT ACTIVE", now)
                self.total_alerts_triggered += 1
                self.emergency_dispatched = True
                event = {
                    "level": "EMERGENCY",
                    "message": "THREAT CONFIRMED: Local alarm activated & simulated emergency payload dispatched!",
                }
            elif fused_threat_score < CAUTION_UPPER_BOUND:
                self._transition_to("RECOVERY", now)
                event = {
                    "level": "INFO",
                    "message": "Distress declining. Entering RECOVERY.",
                }

        elif self.current_state == "ALERT ACTIVE":
            if fused_threat_score < CAUTION_UPPER_BOUND:
                if self._recovery_timer_start is None:
                    self._recovery_timer_start = now
                elif (now - self._recovery_timer_start) >= RECOVERY_CONSECUTIVE_SECS:
                    self._transition_to("RECOVERY", now)
                    event = {
                        "level": "INFO",
                        "message": f"Distress settled below threshold for {RECOVERY_CONSECUTIVE_SECS}s. Recovery initialized.",
                    }
            else:
                self._recovery_timer_start = None

        elif self.current_state == "RECOVERY":
            if fused_threat_score <= SAFE_UPPER_BOUND:
                self._transition_to("SAFE", now)
                self.emergency_dispatched = False
                event = {
                    "level": "INFO",
                    "message": "Full recovery verified. System reset to SAFE baseline.",
                }
            elif fused_threat_score >= THREAT_LOWER_BOUND:
                self._transition_to("THREAT", now)
                event = {
                    "level": "CRITICAL",
                    "message": "Re-escalation during recovery! Returning to THREAT.",
                }

        if event:
            event.update({
                "timestamp": now,
                "old_state": old_state,
                "new_state": self.current_state,
                "threat_score": round(fused_threat_score, 3),
            })

        return event

    def _transition_to(self, new_state: str, timestamp: float) -> None:
        self.current_state = new_state
        self._state_entered_time = timestamp

    def reset_state(self) -> None:
        """Manual reset trigger from UI button."""
        self.current_state = "SAFE"
        self._state_entered_time = time.time()
        self._threat_timer_start = None
        self._recovery_timer_start = None
        self.emergency_dispatched = False

    def get_state_summary(self) -> Dict[str, Any]:
        return {
            "current_state": self.current_state,
            "seconds_in_state": round(time.time() - self._state_entered_time, 1),
            "emergency_dispatched": self.emergency_dispatched,
            "total_alerts": self.total_alerts_triggered,
        }
