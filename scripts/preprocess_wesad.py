"""
Offline WESAD Preprocessing Script.
Scans data/wesad/, extracts normalized sliding windows for all subjects,
fits the global standardizer, and saves training tensors to data/processed/.
"""

import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.config import WESAD_DIR, PROCESSED_DATA_DIR
from backend.data.wesad_adapter import WESADAdapter
from backend.data.preprocessing import SensorPreprocessor


def preprocess_all():
    print("[Preprocessing] Starting WESAD signal preprocessing...")
    adapter = WESADAdapter(WESAD_DIR)
    subjects = adapter.find_available_subjects()

    if not subjects:
        print("[Preprocessing] No WESAD subjects found in data/wesad/.")
        print("[Preprocessing] Generating sample dataset first...")
        from scripts.generate_synthetic_wesad import generate_all_sample_subjects
        generate_all_sample_subjects()
        subjects = adapter.find_available_subjects()

    print(f"[Preprocessing] Processing {len(subjects)} subjects: {subjects}")

    all_raw_signals = []
    subject_data_list = []

    for subj in subjects:
        data = adapter.process_subject(subj)
        if data is not None:
            features = np.stack(
                [data["eda"], data["ecg"], data["respiration"], data["temperature"], data["acc_mag"]],
                axis=-1,
            )
            all_raw_signals.append(features)
            subject_data_list.append(data)
            print(f"  -> Processed {subj}: {len(data['ecg'])} samples")

    # Fit scaler
    preprocessor = SensorPreprocessor()
    preprocessor.fit(all_raw_signals)

    # Extract standardized windows
    all_windows = []
    all_labels = []

    for data in subject_data_list:
        raw_wins, raw_lbls = preprocessor.extract_windows(data)
        if len(raw_wins) > 0:
            std_wins = np.array([preprocessor.transform(w) for w in raw_wins], dtype=np.float32)
            all_windows.append(std_wins)
            all_labels.append(raw_lbls)

    if all_windows:
        X = np.concatenate(all_windows, axis=0)
        y = np.concatenate(all_labels, axis=0)

        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.save(PROCESSED_DATA_DIR / "X_windows.npy", X)
        np.save(PROCESSED_DATA_DIR / "y_labels.npy", y)

        print(f"[Preprocessing] Successfully saved {len(X)} windows to {PROCESSED_DATA_DIR}")
        print(f"  Shape of X: {X.shape}, Shape of y: {y.shape}")
        print(f"  Class Distribution: Baseline (0): {np.sum(y == 0)}, Stress (1): {np.sum(y == 1)}, Amusement (2): {np.sum(y == 2)}")


if __name__ == "__main__":
    preprocess_all()
