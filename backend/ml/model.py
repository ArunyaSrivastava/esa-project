"""
PyTorch LSTM Sequence Classifier for Wearable Distress Detection.
Predicts Baseline (0), Stress (1), and Amusement (2) from multi-channel physiological windows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.config import (
    FEATURE_CHANNELS,
    NUM_CLASSES,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,
)


class SensorLSTMClassifier(nn.Module):
    """
    Lightweight, robust LSTM classifier designed for real-time inference on laptop GPUs/CPUs.
    Architecture:
      Input (B, T, 5) -> LayerNorm -> Linear Projection (64)
      -> 2-Layer LSTM (hidden=64, dropout=0.2)
      -> Last Hidden State -> FC (32) -> ReLU -> Dropout -> FC (3) -> Logits
    """

    def __init__(
        self,
        input_dim: int = len(FEATURE_CHANNELS),
        hidden_dim: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout: float = LSTM_DROPOUT,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Input normalization and feature projection
        self.layer_norm = nn.LayerNorm(input_dim)
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        # Sequence modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classification head
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
          x: Tensor of shape (Batch, Seq_len, Features)
        Returns:
          logits: Tensor of shape (Batch, Num_classes)
        """
        # (B, T, F) -> LayerNorm -> Linear
        x_norm = self.layer_norm(x)
        x_proj = F.relu(self.input_fc(x_norm))

        # LSTM output: out shape (B, T, H), (hn, cn)
        lstm_out, (hn, _) = self.lstm(x_proj)

        # Use last hidden state of final layer
        # hn shape is (num_layers, B, H)
        last_hidden = hn[-1]  # (B, H)

        # Classification logits
        logits = self.fc_head(last_hidden)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Softmax applied."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
