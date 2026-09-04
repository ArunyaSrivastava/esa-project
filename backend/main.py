import asyncio
import base64
import time
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.config import (
    HOST,
    PORT,
    STATIC_DIR,
    WESAD_DIR,
    WEBSOCKET_FPS,
    DEFAULT_SUBJECT,
)
from backend.data.replay_engine import ReplayEngine
from backend.ml.inference import SensorMLPredictor
from backend.vision.camera import CameraManager
from backend.vision.detector import PoseDetector
from backend.vision.pose_features import PoseFeatureAnalyzer
from backend.fusion.fusion_engine import FusionEngine
from backend.fusion.state_machine import AlertStateMachine
from backend.utils.logging import event_logger
from backend.api.routes import router as api_router, set_pipeline_runner
from backend.api.websocket import manager as ws_manager


class MultimodalPipelineRunner:
    """
    Central pipeline orchestrator running the synchronized multi-sensor loop.
    """

    def __init__(self):
        print("==================================================")
        print("   MULTIMODAL THREAT & DISTRESS DETECTION SYSTEM   ")
        print("==================================================")

        # 1. Initialize Sensor Replay Engine (WESAD)
        self._ensure_dataset_present()
        self.replay_engine = ReplayEngine(subject_id=DEFAULT_SUBJECT)
        self.replay_engine.start()

        # 2. Initialize Sensor ML Predictor
        self.sensor_ml = SensorMLPredictor()

        # 3. Initialize Vision Subsystem (Webcam + YOLOv8)
        self.camera = CameraManager()
        self.pose_detector = PoseDetector()
        self.pose_analyzer = PoseFeatureAnalyzer()

        # 4. Initialize Multimodal Fusion & State Machine
        self.fusion_engine = FusionEngine()
        self.state_machine = AlertStateMachine()

        # 5. Demo Scenario Control
        self.active_demo_scenario = "LIVE_FREE"  # LIVE_FREE, NORMAL, PHYSIOLOGICAL_DISTRESS, MULTIMODAL_THREAT, FALSE_POSITIVE_REDUCTION

        # Telemetry & Rates
        self.loop_task: Optional[asyncio.Task] = None
        self._is_running = True
        self._packet_count = 0
        self._fps_counter = 0
        self._last_fps_calc = time.time()
        self._current_fps = 0.0

    def _ensure_dataset_present(self) -> None:
        """Verify data/wesad has at least one subject pickle, else generate samples."""
        has_data = any(WESAD_DIR.glob("S*"))
        if not has_data:
            print("[Init] WESAD dataset not found. Generating sample S2, S3, S4 subjects...")
            from scripts.generate_synthetic_wesad import generate_all_sample_subjects
            generate_all_sample_subjects()

    def set_demo_scenario(self, scenario: str) -> None:
        """Configure demo mode behavior and trigger appropriate sensor/vision state."""
        self.active_demo_scenario = scenario.upper()
        if self.active_demo_scenario == "NORMAL":
            self.replay_engine.seek_to_scenario("BASELINE")
        elif self.active_demo_scenario in ["PHYSIOLOGICAL_DISTRESS", "MULTIMODAL_THREAT", "FALSE_POSITIVE_REDUCTION"]:
            self.replay_engine.seek_to_scenario("STRESS")

    async def run_pipeline_loop(self):
        """Asynchronous high-frequency streaming and fusion loop."""
        interval = 1.0 / WEBSOCKET_FPS
        print(f"[Pipeline] Telemetry streaming loop active at {WEBSOCKET_FPS} FPS.")

        while self._is_running:
            try:
                t0 = time.perf_counter()

                # 1. Process Vision Modality
                raw_frame = self.camera.get_frame()
                detection = self.pose_detector.detect(raw_frame)
                vision_analysis = self.pose_analyzer.analyze(detection, raw_frame)
                annotated_frame = self.pose_analyzer.render_overlay(raw_frame, detection, vision_analysis)

                # 2. Process Sensor Modality (WESAD Replay)
                sensor_packet = self.replay_engine.get_next_sample()
                sensor_window = self.replay_engine.get_current_window()
                sensor_pred = self.sensor_ml.predict_window(sensor_window)
                
                # Advance replay playback based on sampling rate & speed
                samples_to_advance = max(1, int(32 * interval * self.replay_engine.speed))
                self.replay_engine.advance(samples_to_advance)

                # 3. Determine Distress Scores with Demo Scenario Logic
                sensor_distress = sensor_pred["distress_score"]
                vision_distress = vision_analysis["visual_distress_score"]
                demo_override = False
                demo_threat_level = 0.0

                if self.active_demo_scenario == "NORMAL":
                    sensor_distress = 0.08
                    vision_distress = 0.05
                elif self.active_demo_scenario == "PHYSIOLOGICAL_DISTRESS":
                    sensor_distress = 0.85
                    vision_distress = 0.08
                elif self.active_demo_scenario == "MULTIMODAL_THREAT":
                    sensor_distress = 0.88
                    vision_distress = 0.82
                elif self.active_demo_scenario == "FALSE_POSITIVE_REDUCTION":
                    sensor_distress = 0.90
                    vision_distress = 0.06

                # 4. Multimodal Fusion
                fusion_result = self.fusion_engine.fuse(
                    sensor_distress=sensor_distress,
                    vision_distress=vision_distress,
                    demo_override=demo_override,
                    demo_threat_level=demo_threat_level,
                )

                # 5. Alert State Machine Evaluation
                state_event = self.state_machine.update(fusion_result["threat_score"])
                if state_event:
                    event_logger.log_event(
                        category="ALERT",
                        message=state_event["message"],
                        level=state_event["level"],
                        meta=state_event,
                    )

                # 6. Encode Annotated Frame to JPEG Base64
                _, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_b64 = base64.b64encode(buffer).decode("utf-8")

                # 7. Update Telemetry Counters
                self._packet_count += 1
                self._fps_counter += 1
                now = time.time()
                if now - self._last_fps_calc >= 1.0:
                    self._current_fps = round(self._fps_counter / (now - self._last_fps_calc), 1)
                    self._fps_counter = 0
                    self._last_fps_calc = now

                # 8. Construct WebSocket Payload
                payload = {
                    "type": "TELEMETRY",
                    "timestamp": now,
                    "fps": self._current_fps,
                    "packet_counter": self._packet_count,
                    "demo_scenario": self.active_demo_scenario,
                    "camera_frame": f"data:image/jpeg;base64,{frame_b64}",
                    "sensor_packet": sensor_packet.to_dict() if sensor_packet else {},
                    "sensor_ml": {
                        "distress_score": sensor_distress,
                        "distress_pct": round(sensor_distress * 100, 1),
                        "probabilities": sensor_pred["probabilities"],
                        "predicted_label": sensor_pred["predicted_label"],
                        "latency_ms": sensor_pred["latency_ms"],
                    },
                    "vision_ml": {
                        "distress_score": vision_distress,
                        "distress_pct": round(vision_distress * 100, 1),
                        "posture": vision_analysis["posture"],
                        "facial_expression": vision_analysis.get("facial_expression", "Neutral"),
                        "is_smiling": bool(vision_analysis.get("is_smiling", False)),
                        "movement_level": vision_analysis["movement_level"],
                        "motion_velocity": vision_analysis["motion_velocity"],
                        "confidence": vision_analysis["confidence"],
                        "person_detected": vision_analysis["person_detected"],
                        "is_closeup": vision_analysis.get("is_closeup", False),
                    },
                    "fusion": fusion_result,
                    "state_machine": self.state_machine.get_state_summary(),
                    "replay": self.replay_engine.get_current_state(),
                    "recent_logs": event_logger.get_recent_logs(20),
                }

                # 9. Broadcast to Connected Web Clients
                await ws_manager.broadcast_json(payload)

                # Pacing sleep
                elapsed = time.perf_counter() - t0
                sleep_time = max(0.001, interval - elapsed)
                await asyncio.sleep(sleep_time)

            except Exception as e:
                print(f"[Pipeline Loop Error] {e}")
                await asyncio.sleep(0.1)


# FastAPI Application Setup
app = FastAPI(title="Multimodal IoT Threat Detection System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

runner: Optional[MultimodalPipelineRunner] = None


@app.on_event("startup")
async def startup_event():
    global runner
    runner = MultimodalPipelineRunner()
    set_pipeline_runner(runner)
    runner.loop_task = asyncio.create_task(runner.run_pipeline_loop())


@app.on_event("shutdown")
async def shutdown_event():
    global runner
    if runner:
        runner._is_running = False
        runner.camera.stop()
        runner.replay_engine.stop()


# Mount REST API
app.include_router(api_router)


# WebSocket Route
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Client can send commands via WS as well
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


# Mount Static Frontend
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="frontend")


def main():
    print(f"Starting Personal Safety Monitor Dashboard on http://localhost:{PORT} ...")
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=False, log_level="warning")


if __name__ == "__main__":
    main()
