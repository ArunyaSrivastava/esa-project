"""
Sensor Signal Preprocessing, Standardization & Windowing Pipeline.
Prepares continuous multi-channel signals for PyTorch sequence modeling.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from backend.config import (
    SCALER_PATH,
    WINDOW_SAMPLES,
    FEATURE_CHANNELS,
    PROCESSED_DATA_DIR,
)


class SensorPreprocessor:
    """
    Standardizes physiological signals and constructs windowed tensors.
    """

    def __init__(self, scaler_path: Path = SCALER_PATH):
        self.scaler_path = Path(scaler_path)
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.load_scaler()

    def fit(self, signals_list: List[np.ndarray]) -> None:
        """
        Compute mean and standard deviation across all training signals.
        Each signal in signals_list is shape (N_samples, N_features).
        """
        all_data = np.concatenate(signals_list, axis=0)
        self.mean = np.nanmean(all_data, axis=0).astype(np.float32)
        self.std = np.nanstd(all_data, axis=0).astype(np.float32)
        # Avoid division by zero
        self.std = np.where(self.std < 1e-5, 1.0, self.std)
        self.save_scaler()

    def transform(self, window: np.ndarray) -> np.ndarray:
        """
        Standardize a window (N_samples, N_features) using fitted mean/std.
        Fills any NaN values with zeros.
        """
        clean_window = np.nan_to_num(window, nan=0.0)
        if self.mean is not None and self.std is not None:
            return (clean_window - self.mean) / self.std
        return clean_window

    def save_scaler(self) -> None:
        """Save scaler statistics to JSON."""
        if self.mean is not None and self.std is not None:
            data = {
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
                "features": FEATURE_CHANNELS,
            }
            self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.scaler_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[Preprocessor] Saved scaler params to {self.scaler_path}")

    def load_scaler(self) -> bool:
        """Load scaler statistics if available."""
        if self.scaler_path.exists():
            try:
                with open(self.scaler_path, "r") as f:
                    data = json.load(f)
                self.mean = np.array(data["mean"], dtype=np.float32)
                self.std = np.array(data["std"], dtype=np.float32)
                return True
            except Exception as e:
                print(f"[Preprocessor] Error loading scaler: {e}")
        return False

    @staticmethod
    def extract_windows(
        subject_data: Dict[str, np.ndarray],
        window_size: int = WINDOW_SAMPLES,
        step_size: int = WINDOW_SAMPLES // 4,  # 75% overlap for rich training sets
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract sliding windows of shape (N_windows, window_size, 5)
        and corresponding labels (N_windows,).
        """
        eda = subject_data["eda"]
        ecg = subject_data["ecg"]
        resp = subject_data["respiration"]
        temp = subject_data["temperature"]
        acc_mag = subject_data["acc_mag"]
        labels = subject_data["label"]

        features = np.stack([eda, ecg, resp, temp, acc_mag], axis=-1)  # (T, 5)
        total_len = len(labels)

        windows = []
        window_labels = []

        for start_idx in range(0, total_len - window_size + 1, step_size):
            end_idx = start_idx + window_size
            win_feat = features[start_idx:end_idx]
            win_lbl = labels[start_idx:end_idx]

            # Use majority valid label (0, 1, 2) in window
            valid_lbls = win_lbl[win_lbl >= 0]
            if len(valid_lbls) >= (window_size * 0.5):
                counts = np.bincount(valid_lbls)
                maj_label = int(np.argmax(counts))
                windows.append(win_feat)
                window_labels.append(maj_label)

        if len(windows) == 0:
            return np.empty((0, window_size, 5), dtype=np.float32), np.empty((0,), dtype=np.int64)

        return np.array(windows, dtype=np.float32), np.array(window_labels, dtype=np.int64)
