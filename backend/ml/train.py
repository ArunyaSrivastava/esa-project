"""
Model Training Routine for PyTorch Sensor LSTM Classifier.
Includes automated data checking, train/val split, loss tracking,
confusion matrix generation, and model artifact checkpointing.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.config import (
    MODEL_PATH,
    CONFUSION_MATRIX_PATH,
    PROCESSED_DATA_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    LABEL_NAMES,
)
from backend.ml.model import SensorLSTMClassifier
from backend.ml.dataset import SensorSequenceDataset
from scripts.preprocess_wesad import preprocess_all


def train_sensor_model(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE):
    print("==================================================")
    print("       STARTING SENSOR LSTM MODEL TRAINING        ")
    print("==================================================")

    # Check for preprocessed data
    x_path = PROCESSED_DATA_DIR / "X_windows.npy"
    y_path = PROCESSED_DATA_DIR / "y_labels.npy"

    if not x_path.exists() or not y_path.exists():
        print("[Train] Processed datasets not found. Running preprocessor...")
        preprocess_all()

    X = np.load(x_path)
    y = np.load(y_path)

    print(f"[Train] Loaded dataset with {len(X)} windows. Shape: {X.shape}")

    # Create PyTorch dataset and splits
    dataset = SensorSequenceDataset(X, y)
    val_size = max(int(0.2 * len(dataset)), 1)
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Hardware device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Training on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Instantiate model, optimizer, loss
    model = SensorLSTMClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_y)

        train_loss /= train_size

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * len(batch_y)

                preds = torch.argmax(outputs, dim=-1)
                correct += (preds == batch_y).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        val_loss /= val_size
        val_acc = correct / val_size

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "epoch": epoch,
                    "input_dim": model.input_dim,
                    "hidden_dim": model.hidden_dim,
                },
                MODEL_PATH,
            )

    print(f"[Train] Model saved to {MODEL_PATH} (Best Val Accuracy: {best_val_acc * 100:.2f}%)")

    # Generate Confusion Matrix
    _save_confusion_matrix(all_targets, all_preds, history)
    print("==================================================")
    print("           TRAINING COMPLETE & SAVED              ")
    print("==================================================")
    return best_val_acc


def _save_confusion_matrix(targets, preds, history):
    """Plot and save confusion matrix and training curves using pure numpy."""
    try:
        import itertools

        targets_arr = np.array(targets, dtype=np.int32)
        preds_arr = np.array(preds, dtype=np.int32)
        
        # 3x3 confusion matrix with pure numpy
        cm = np.zeros((3, 3), dtype=np.int32)
        for t, p in zip(targets_arr, preds_arr):
            if 0 <= t < 3 and 0 <= p < 3:
                cm[t, p] += 1

        classes = [LABEL_NAMES[i] for i in range(3)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
        fig.patch.set_facecolor("#111111")
        
        # Subplot 1: Confusion Matrix
        ax1.set_facecolor("#111111")
        im = ax1.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax1.set_title("Confusion Matrix", color="#ccff00", fontsize=12, fontweight="bold")
        tick_marks = np.arange(len(classes))
        ax1.set_xticks(tick_marks)
        ax1.set_xticklabels(classes, color="#e0e0e0")
        ax1.set_yticks(tick_marks)
        ax1.set_yticklabels(classes, color="#e0e0e0")

        thresh = cm.max() / 2.0 if cm.max() > 0 else 1
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax1.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

        ax1.set_ylabel("True Label", color="#00f0ff")
        ax1.set_xlabel("Predicted Label", color="#00f0ff")

        # Subplot 2: Loss & Accuracy
        ax2.set_facecolor("#111111")
        ax2.plot(history["train_loss"], label="Train Loss", color="#00f0ff", lw=2)
        ax2.plot(history["val_loss"], label="Val Loss", color="#ff003c", lw=2)
        ax2.set_title("Training Loss & Convergence", color="#ccff00", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch", color="#e0e0e0")
        ax2.set_ylabel("Loss", color="#e0e0e0")
        ax2.tick_params(colors="#e0e0e0")
        ax2.legend(facecolor="#1e1e1e", edgecolor="#2a2a2a", labelcolor="#e0e0e0")
        ax2.grid(True, color="#2a2a2a", linestyle="--", alpha=0.5)

        plt.tight_layout()
        CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(CONFUSION_MATRIX_PATH, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close()
        print(f"[Train] Saved confusion matrix & curves to {CONFUSION_MATRIX_PATH}")
    except Exception as e:
        print(f"[Train] Error plotting confusion matrix: {e}")


if __name__ == "__main__":
    train_sensor_model()
