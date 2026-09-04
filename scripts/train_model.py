"""
Standalone Sensor Model Training Entrypoint.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.ml.train import train_sensor_model

if __name__ == "__main__":
    train_sensor_model()
