"""
REST API Endpoints for System Control and Auditing.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any

from backend.utils.logging import event_logger

router = APIRouter(prefix="/api")


class ReplayControlRequest(BaseModel):
    action: str  # play, pause, reset, speed, seek, subject, scenario
    speed: Optional[float] = None
    seek_fraction: Optional[float] = None
    subject_id: Optional[str] = None
    scenario: Optional[str] = None


class WeightConfigRequest(BaseModel):
    sensor_weight: float
    vision_weight: float


class DemoScenarioRequest(BaseModel):
    scenario: str  # NORMAL, PHYSIOLOGICAL_DISTRESS, MULTIMODAL_THREAT, FALSE_POSITIVE_REDUCTION, LIVE_FREE


# Global reference to pipeline runner (injected at startup)
pipeline_runner = None


def set_pipeline_runner(runner):
    global pipeline_runner
    pipeline_runner = runner


@router.post("/replay/control")
def control_replay(req: ReplayControlRequest):
    if not pipeline_runner:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    rep = pipeline_runner.replay_engine
    act = req.action.lower()

    if act == "play":
        rep.start()
        event_logger.log_event("REPLAY", "Sensor replay started.")
    elif act == "pause":
        rep.stop()
        event_logger.log_event("REPLAY", "Sensor replay paused.")
    elif act == "reset":
        rep.reset()
        event_logger.log_event("REPLAY", "Sensor replay reset to start.")
    elif act == "speed" and req.speed is not None:
        rep.set_speed(req.speed)
        event_logger.log_event("REPLAY", f"Replay speed set to {req.speed}x")
    elif act == "seek" and req.seek_fraction is not None:
        rep.seek_fraction(req.seek_fraction)
        event_logger.log_event("REPLAY", f"Seeked to {int(req.seek_fraction*100)}%")
    elif act == "subject" and req.subject_id:
        success = rep.load_subject(req.subject_id)
        if success:
            event_logger.log_event("REPLAY", f"Loaded subject {req.subject_id}")
        else:
            raise HTTPException(status_code=400, detail=f"Subject {req.subject_id} could not be loaded")
    elif act == "scenario" and req.scenario:
        success = rep.seek_to_scenario(req.scenario)
        event_logger.log_event("REPLAY", f"Jumped to scenario segment: {req.scenario}")

    return {"status": "success", "replay_state": rep.get_current_state()}


@router.get("/replay/subjects")
def get_subjects():
    if not pipeline_runner:
        return {"subjects": ["S2", "S3", "S4"]}
    return {"subjects": pipeline_runner.replay_engine.get_available_subjects()}


@router.post("/fusion/weights")
def set_weights(req: WeightConfigRequest):
    if not pipeline_runner:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    pipeline_runner.fusion_engine.set_weights(req.sensor_weight, req.vision_weight)
    event_logger.log_event("FUSION", f"Weights adjusted: Sensor={req.sensor_weight:.2f}, Vision={req.vision_weight:.2f}")
    return {"status": "success", "sensor_weight": pipeline_runner.fusion_engine.sensor_weight, "vision_weight": pipeline_runner.fusion_engine.vision_weight}


@router.post("/demo/scenario")
def trigger_demo_scenario(req: DemoScenarioRequest):
    if not pipeline_runner:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    scenario = req.scenario.upper()
    pipeline_runner.set_demo_scenario(scenario)
    event_logger.log_event("DEMO_MODE", f"Active Scenario: {scenario}", level="WARNING")
    return {"status": "success", "active_scenario": scenario}


@router.post("/alert/reset")
def reset_alert():
    if not pipeline_runner:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    pipeline_runner.state_machine.reset_state()
    event_logger.log_event("ALERT", "Manual Alert Reset executed by operator.", level="INFO")
    return {"status": "success", "alert_state": pipeline_runner.state_machine.get_state_summary()}


@router.get("/logs")
def get_logs():
    return {"logs": event_logger.get_recent_logs(60)}


@router.get("/logs/export/json")
def export_json_logs():
    fpath = event_logger.export_json()
    return FileResponse(fpath, media_type="application/json", filename=fpath.name)


@router.get("/logs/export/csv")
def export_csv_logs():
    fpath = event_logger.export_csv()
    return FileResponse(fpath, media_type="text/csv", filename=fpath.name)


@router.get("/status")
def get_system_status():
    return {
        "status": "ONLINE",
        "data_source": {
            "wearable": f"WESAD Replay ({pipeline_runner.replay_engine.subject_id if pipeline_runner else 'S2'})",
            "camera": "Live Laptop Webcam" if (pipeline_runner and pipeline_runner.camera.is_camera_live) else "Synthetic Fallback Pattern",
            "hardware": "Not Connected (ESP32 Ready)",
        },
        "models": {
            "sensor_model": "PyTorch 2-Layer LSTM Classifier (Trained)",
            "vision_model": "YOLOv8n-Pose Keypoint Model (CUDA Accelerated)",
            "fusion": "Weighted Multimodal Decision Engine",
        }
    }
