"""
WESAD Dataset Adapter.
Parses official WESAD .pkl files and normalizes signals into a unified sampling rate.
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from scipy import signal
from scipy.signal import find_peaks

from backend.config import (
    WESAD_DIR,
    SENSOR_SAMPLING_RATE_HZ,
    FEATURE_CHANNELS,
)


class WESADAdapter:
    """
    Adapter for loading and parsing real WESAD subject pickle files.
    Standardizes raw 700Hz chest and variable wrist signals to a unified rate.
    """

    # Raw WESAD label mapping to our 3-class system
    # Raw: 1 = Baseline, 2 = Stress, 3 = Amusement, 0/4/5/6/7 = Other/Transient
    LABEL_MAP = {
        1: 0,  # Baseline -> Class 0
        2: 1,  # Stress   -> Class 1
        3: 2,  # Amusement-> Class 2
    }

    def __init__(self, data_dir: Path = WESAD_DIR):
        self.data_dir = Path(data_dir)

    def find_available_subjects(self) -> List[str]:
        """Scan directory for available WESAD subject files (S2, S3, etc.)."""
        if not self.data_dir.exists():
            return []
        
        subjects = []
        for p in self.data_dir.glob("S*"):
            if p.is_dir() and (p / f"{p.name}.pkl").exists():
                subjects.append(p.name)
            elif p.suffix == ".pkl":
                subjects.append(p.stem)
        
        return sorted(list(set(subjects)))

    def load_subject_raw(self, subject_id: str) -> Optional[Dict[str, Any]]:
        """Load raw pickle file for a subject with encoding compatibility."""
        candidates = [
            self.data_dir / subject_id / f"{subject_id}.pkl",
            self.data_dir / f"{subject_id}.pkl",
        ]
        
        target_path = None
        for cand in candidates:
            if cand.exists():
                target_path = cand
                break

        if not target_path:
            return None

        try:
            with open(target_path, "rb") as f:
                # WESAD pickles were generated with Python 2/3 latin1 encoding
                data = pickle.load(f, encoding="latin1")
            return data
        except Exception as e:
            print(f"[WESADAdapter] Error reading {target_path}: {e}")
            return None

    def process_subject(self, subject_id: str, target_hz: int = SENSOR_SAMPLING_RATE_HZ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract chest & wrist signals, downsample from 700Hz to target_hz (e.g. 32Hz),
        and calculate derived metrics (Heart Rate, Accel Magnitude).
        """
        raw_data = self.load_subject_raw(subject_id)
        if raw_data is None:
            return None

        try:
            chest_data = raw_data["signal"]["chest"]
            raw_labels = raw_data["label"].flatten()
            orig_len = len(raw_labels)
            raw_hz = 700
            
            # Downsample factor
            downsample_factor = raw_hz // target_hz
            target_len = orig_len // downsample_factor

            # Extract raw chest signals
            raw_ecg = chest_data["ECG"].flatten()[:target_len * downsample_factor]
            raw_eda = chest_data["EDA"].flatten()[:target_len * downsample_factor]
            raw_resp = chest_data["Resp"].flatten()[:target_len * downsample_factor]
            raw_temp = chest_data["Temp"].flatten()[:target_len * downsample_factor]
            raw_acc = chest_data["ACC"][:target_len * downsample_factor, :]
            raw_labels_cut = raw_labels[:target_len * downsample_factor]

            # Resample / decimate using block averaging for stability & speed
            ecg_resampled = signal.decimate(raw_ecg, downsample_factor, zero_phase=True)
            eda_resampled = signal.decimate(raw_eda, downsample_factor, zero_phase=True)
            resp_resampled = signal.decimate(raw_resp, downsample_factor, zero_phase=True)
            temp_resampled = signal.decimate(raw_temp, downsample_factor, zero_phase=True)
            
            acc_x = signal.decimate(raw_acc[:, 0], downsample_factor, zero_phase=True)
            acc_y = signal.decimate(raw_acc[:, 1], downsample_factor, zero_phase=True)
            acc_z = signal.decimate(raw_acc[:, 2], downsample_factor, zero_phase=True)
            acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

            # Downsample labels by mode / nearest subsampling
            labels_resampled = raw_labels_cut[::downsample_factor][:len(ecg_resampled)]
            mapped_labels = np.array([self.LABEL_MAP.get(lbl, -1) for lbl in labels_resampled], dtype=np.int32)

            # Compute estimated heart rate (BPM) from ECG with rolling window
            heart_rates = self._compute_heart_rate(ecg_resampled, target_hz)

            processed = {
                "subject_id": subject_id,
                "ecg": ecg_resampled.astype(np.float32),
                "eda": eda_resampled.astype(np.float32),
                "respiration": resp_resampled.astype(np.float32),
                "temperature": temp_resampled.astype(np.float32),
                "acc_x": acc_x.astype(np.float32),
                "acc_y": acc_y.astype(np.float32),
                "acc_z": acc_z.astype(np.float32),
                "acc_mag": acc_mag.astype(np.float32),
                "heart_rate": heart_rates.astype(np.float32),
                "label": mapped_labels,
                "sampling_rate": target_hz,
            }
            return processed
        except Exception as e:
            print(f"[WESADAdapter] Error processing subject {subject_id}: {e}")
            return None

    def _compute_heart_rate(self, ecg_signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """Estimate instantaneous heart rate across time from ECG peaks."""
        # Normalize ECG
        ecg_norm = (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-6)
        
        # Min distance between R-peaks at 32Hz (e.g. max 200 BPM = 0.3s = ~10 samples)
        min_dist = max(int(0.3 * sample_rate), 5)
        peaks, _ = find_peaks(ecg_norm, distance=min_dist, height=0.5)

        hr_bpm = np.full_like(ecg_signal, 72.0)  # Default 72 BPM
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks) / sample_rate  # Seconds between peaks
            bpm_values = 60.0 / np.clip(peak_intervals, 0.3, 2.0)  # Clip to 30 - 200 BPM
            
            # Interpolate BPM across all time steps
            for i in range(len(peaks) - 1):
                p_start = peaks[i]
                p_end = peaks[i + 1]
                hr_bpm[p_start:p_end] = bpm_values[i]
            
            if peaks[0] > 0:
                hr_bpm[:peaks[0]] = bpm_values[0]
            if peaks[-1] < len(ecg_signal):
                hr_bpm[peaks[-1]:] = bpm_values[-1]

        return hr_bpm
