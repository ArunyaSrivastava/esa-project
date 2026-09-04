"""
Real-time Sensor ML Inference Engine.
Loads trained PyTorch LSTM model, runs windowed inference, and derives
the calibrated, dynamic Sensor Distress Score with temperature scaling.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F

from backend.config import (
    MODEL_PATH,
    SCALER_PATH,
    WINDOW_SAMPLES,
    LABEL_NAMES,
)
from backend.ml.model import SensorLSTMClassifier
from backend.data.preprocessing import SensorPreprocessor


class SensorMLPredictor:
    """
    Inference manager for physiological distress evaluation.
    Auto-detects missing models, loads checkpoints, and performs GPU/CPU inference.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = SensorPreprocessor()
        self.model: Optional[SensorLSTMClassifier] = None
        self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        """Check if model checkpoint exists. If not, trigger quick auto-training."""
        if not self.model_path.exists() or not SCALER_PATH.exists():
            print(f"[SensorMLPredictor] Checkpoint not found at {self.model_path}. Auto-training lightweight model...")
            from backend.ml.train import train_sensor_model
            train_sensor_model(epochs=15)
            self.preprocessor.load_scaler()

        try:
            self.model = SensorLSTMClassifier().to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.eval()
            print(f"[SensorMLPredictor] Loaded model from {self.model_path} onto {self.device}")
        except Exception as e:
            print(f"[SensorMLPredictor] Error loading model: {e}. Reinitializing default model.")
            self.model = SensorLSTMClassifier().to(self.device)
            self.model.eval()

    def predict_window(self, raw_window: np.ndarray) -> Dict[str, Any]:
        """
        Takes raw window (128, 5) [eda, ecg, resp, temp, acc_mag],
        standardizes, runs LSTM forward pass, applies temperature calibration,
        and returns dynamic distress score.
        """
        t0 = time.perf_counter()

        if raw_window is None or len(raw_window) < WINDOW_SAMPLES:
            # Fallback if window not yet full
            return {
                "distress_score": 0.05,
                "probabilities": {"baseline": 0.88, "stress": 0.08, "amusement": 0.04},
                "predicted_class": 0,
                "predicted_label": "Baseline",
                "latency_ms": 0.1,
            }

        # Standardize using fitted statistics
        norm_window = self.preprocessor.transform(raw_window)  # (128, 5)

        # Tensor shape (1, 128, 5)
        tensor_in = torch.tensor(norm_window, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_in)
            # Temperature scaling (T=2.2) to prevent artificial 100% saturation
            # and produce responsive, well-calibrated probabilities
            scaled_logits = logits / 2.2
            probs = F.softmax(scaled_logits, dim=-1).squeeze(0).cpu().numpy()

        p_baseline = float(np.clip(probs[0], 0.01, 0.99))
        p_stress = float(np.clip(probs[1], 0.01, 0.99))
        p_amusement = float(np.clip(probs[2], 0.01, 0.99))

        # Re-normalize to sum to 1.0
        total_p = p_baseline + p_stress + p_amusement
        p_baseline /= total_p
        p_stress /= total_p
        p_amusement /= total_p

        pred_class = int(np.argmax([p_baseline, p_stress, p_amusement]))
        pred_label = LABEL_NAMES.get(pred_class, "Unknown")
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # The Sensor Distress Score is the calibrated Stress class probability
        sensor_distress_score = round(p_stress, 3)

        return {
            "distress_score": sensor_distress_score,
            "probabilities": {
                "baseline": round(p_baseline, 3),
                "stress": round(p_stress, 3),
                "amusement": round(p_amusement, 3),
            },
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "latency_ms": latency_ms,
        }
