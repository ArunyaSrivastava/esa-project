"""
Centralized Configuration for Multimodal Threat & Distress Detection System.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
WESAD_DIR = DATA_DIR / "wesad"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "frontend"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for path in [DATA_DIR, WESAD_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Sensor & Replay Settings
DEFAULT_SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11"]
DEFAULT_SUBJECT = "S2"
DEFAULT_REPLAY_SPEED = 1.0  # 0.25x, 0.5x, 1.0x, 2.0x, 5.0x
SENSOR_SAMPLING_RATE_HZ = 32  # Unified resampled rate for replay & model windowing
WINDOW_SIZE_SECONDS = 4  # 4 seconds per sub-window for real-time inference (128 samples at 32Hz)
WINDOW_STEP_SECONDS = 1  # 1 second step
WINDOW_SAMPLES = SENSOR_SAMPLING_RATE_HZ * WINDOW_SIZE_SECONDS  # 128 samples

# Sensor Features
FEATURE_CHANNELS = ["eda", "ecg", "respiration", "temperature", "acc_mag"]
NUM_CLASSES = 3  # 0: Baseline, 1: Stress, 2: Amusement
LABEL_NAMES = {0: "Baseline", 1: "Stress", 2: "Amusement"}

# ML Model Settings
MODEL_PATH = MODELS_DIR / "lstm_sensor_model.pt"
SCALER_PATH = MODELS_DIR / "scaler_params.json"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 20

# Vision Subsystem Settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
YOLO_MODEL_NAME = "yolov8n-pose.pt"  # Lightweight YOLOv8 pose model
POSE_CONF_THRESHOLD = 0.4
FALL_ASPECT_RATIO_THRESH = 0.75  # Height-to-width ratio indicating horizontal body orientation
DEFENSIVE_WRIST_HEAD_THRESH = 0.35  # Normalized distance of wrists to nose/ears
RAPID_MOTION_THRESH = 1.5  # Standardized keypoint velocity magnitude

# Multimodal Fusion Settings
SENSOR_WEIGHT = 0.60
VISION_WEIGHT = 0.40

# Decision Thresholds
SAFE_UPPER_BOUND = 0.39
CAUTION_UPPER_BOUND = 0.69
THREAT_LOWER_BOUND = 0.70

# Hysteresis & Temporal Consistency
THREAT_CONSECUTIVE_SECS = 2.5  # Must maintain elevated score for 2.5s before THREAT
RECOVERY_CONSECUTIVE_SECS = 3.0  # Must drop below threshold for 3.0s before SAFE
ROLLING_WINDOW_SIZE = 5  # Number of recent inference samples to average

# Server Settings
HOST = "0.0.0.0"
PORT = 8000
WEBSOCKET_FPS = 20
