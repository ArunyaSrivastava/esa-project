"""
PyTorch Dataset for Physiological Sequences.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class SensorSequenceDataset(Dataset):
    """
    Dataset wrapper for windowed physiological signals.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
