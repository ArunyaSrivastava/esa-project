# Multimodal IoT-Based Threat & Distress Detection System
### College Embedded Systems Capstone / Prototype Demonstration

A real-time multimodal safety monitoring system that fuses wearable physiological sensor streams (emulated via the **WESAD** dataset) with live laptop-camera **Computer Vision** (YOLOv8-pose) to detect physical distress, struggle, and threats in real time.

Designed with a clean, modular architecture that allows the current WESAD replay source to be replaced with an **ESP32 wearable prototype** (MAX30102 PPG + MPU6050 6-axis IMU) without modifying the downstream ML, vision, fusion, or UI layers.

---

## Key System Highlights

- **Multimodal Context-Aware Fusion**: A single high heart rate or sudden motion does not trigger an alarm. Threat decisions require cross-modal agreement and temporal persistence, drastically reducing false positives.
- **Sensor Sequence ML Model**: PyTorch 2-layer LSTM sequence classifier predicting Baseline, Stress, and Amusement from multi-channel physiological signals (EDA, ECG/HR, Respiration, Skin Temperature, Acceleration Magnitude).
- **Vision Subsystem**: Real-time webcam capture with YOLOv8-pose keypoint estimation detecting falls, struggle motions, and raised defensive postures.
- **Deterministic Demo Scenarios**: Built-in review scenarios (Normal Baseline, Stress Suppression, Multimodal Threat, False Alarm Reduction) for bulletproof review demonstrations.
- **UniArchives-Matched Cyber-Industrial Dashboard**: Futuristic UI aesthetic styled after the UniArchives design system (JetBrains Mono & Unbounded typography, `#050505`/`#111111` dark theme, `#ccff00` neon green, `#00f0ff` cyan, `#ff003c` alert red, hard-edge shadows, real-time 60 FPS waveform canvases, and live video overlays).

---

## System Architecture

```
                                 ┌────────────────────────┐
                                 │   WESAD Dataset .pkl   │
                                 │ (S2, S3, S4 Subjects)  │
                                 └───────────┬────────────┘
                                             │
                                             ▼
                                 ┌────────────────────────┐
                                 │   WESADSensorSource    │
                                 │ (SensorSource Interface│
                                 └───────────┬────────────┘
                                             │
  ┌───────────────────────┐                  ▼
  │  Future Physical      │      ┌────────────────────────┐
  │  ESP32 Wearable       ├─────►│     Replay Engine      │
  │  (MAX30102 + MPU6050) │      │(Play/Pause/Speed/Seek) │
  └───────────────────────┘      └───────────┬────────────┘
                                             │
                                             ▼
                                 ┌────────────────────────┐
                                 │ Signal Preprocessing   │
                                 │ (Resample, Standardize)│
                                 └───────────┬────────────┘
                                             │
                                             ▼
                                 ┌────────────────────────┐
                                 │  PyTorch 2-Layer LSTM  │
                                 │  Classifier (3-Class)  │
                                 └───────────┬────────────┘
                                             │
                                             ▼
                                   [Sensor Distress Score]
                                             │
                                             ├───────────────────────────────────┐
                                             │                                   │
                                             ▼                                   ▼
┌─────────────────────────┐      ┌────────────────────────┐         ┌─────────────────────────┐
│   Laptop Webcam Feed    │      │   MULTIMODAL FUSION    │         │  UniArchives Cyber UI   │
│   (OpenCV DirectShow)   │      │         ENGINE         │         │  (Live Video & Pose,    │
└────────────┬────────────┘      │ (Weighted + Consensus) │         │   Waveforms, Gauges,    │
             │                   └───────────┬────────────┘         │   Audit Event Log)      │
             ▼                               │                      └─────────────────────────┘
┌─────────────────────────┐                  ▼                                   ▲
│  YOLOv8-Pose Detector   │         [Final Threat Score]                         │
│  (CUDA Keypoints)       │                  │                                   │
└────────────┬────────────┘                  ▼                                   │
             │                   ┌────────────────────────┐                      │
             ▼                   │   Alert State Machine  │                      │
┌─────────────────────────┐      │(SAFE/CAUTION/THREAT/   ├──────────────────────┘
│ Pose & Motion Analyzer  │      │ ALERT ACTIVE/RECOVERY) │
│ (Fall/Struggle/Defense) │      └────────────────────────┘
└────────────┬────────────┘
             │
             ▼
   [Visual Distress Score]
```

---

## Directory Structure

```
esa_project/
├── backend/
│   ├── config.py                 # Central configuration (weights, thresholds, rates)
│   ├── main.py                   # FastAPI server, WebSocket hub, pipeline loop
│   ├── data/
│   │   ├── sensor_source.py      # Abstract SensorSource interface & NormalizedPacket
│   │   ├── wesad_adapter.py      # WESAD .pkl parser, signal decimation & HR derivation
│   │   ├── replay_engine.py      # Deterministic replayer (0.25x-5x speed, loop, seek)
│   │   ├── esp32_adapter.py      # Future ESP32 serial/TCP connector stub
│   │   └── preprocessing.py      # Resampling, windowing, standardization
│   ├── ml/
│   │   ├── dataset.py            # PyTorch Dataset wrapper
│   │   ├── model.py              # PyTorch 2-Layer LSTM Classifier
│   │   ├── train.py              # Automated training routine & confusion matrix
│   │   └── inference.py          # Real-time GPU/CPU predictor
│   ├── vision/
│   │   ├── camera.py             # Threaded OpenCV webcam capture & test fallback
│   │   ├── detector.py           # YOLOv8n-pose detection wrapper
│   │   └── pose_features.py      # Fall detection, struggle motion, cyber HUD overlay
│   ├── fusion/
│   │   ├── fusion_engine.py      # Weighted fusion with false-positive suppression
│   │   └── state_machine.py      # Finite alert state machine with hysteresis
│   ├── api/
│   │   ├── routes.py             # REST API (replay control, scenarios, export)
│   │   └── websocket.py          # WebSocket client manager
│   └── utils/
│       └── logging.py            # Timestamped safety event logger (JSON/CSV export)
├── frontend/
│   ├── index.html                # UniArchives-styled cyber-industrial dashboard
│   ├── css/
│   │   └── style.css             # Cyber styling, neon accents, hard shadows
│   └── js/
│       ├── app.js                # WebSocket client, Web Audio alarm, controls
│       └── charts.js             # High-performance 60 FPS HTML5 canvas charts
├── data/
│   ├── wesad/                    # WESAD subject pickle files (S2, S3, S4)
│   └── processed/                # Preprocessed sliding window tensors
├── models/
│   ├── lstm_sensor_model.pt      # Trained PyTorch LSTM checkpoint
│   ├── scaler_params.json        # Signal standardizer parameters
│   └── confusion_matrix.png      # Validation confusion matrix plot
├── scripts/
│   ├── download_wesad.py         # Downloader for official 2GB WESAD archive
│   ├── generate_synthetic_wesad.py # High-fidelity WESAD sample generator
│   ├── preprocess_wesad.py       # Offline signal preprocessor
│   └── train_model.py            # Standalone training script
├── run.py                        # Single-command application launcher
├── requirements.txt              # Project dependencies
├── .env.example                  # Environment configuration example
└── README.md                     # Comprehensive documentation
```

---

## Quickstart & Installation

### 1. Prerequisites
- Python 3.10+
- Webcam (built-in or USB)
- NVIDIA GPU with CUDA (optional; CPU inference is fully supported)

### 2. Setup Environment
```bash
# Clone or navigate to the repository
cd "c:\Users\DELL\Documents\esa project"

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Application (Single Command)
```bash
python run.py
```
Open your browser and navigate to:
```
http://localhost:8000
```

> **Note**: On first launch, the system automatically verifies whether WESAD data and trained models exist. If not, it generates high-fidelity sample subjects, trains the PyTorch LSTM model in seconds, downloads the lightweight YOLOv8-pose model, and starts the live server.

---

## Dataset & Emulation Strategy

### How WESAD Simulates the IoT Wearable
In the final product, the wearable transmits raw sensor readings over WiFi/Bluetooth. For this demonstration, the `ReplayEngine` streams pre-recorded physiological data from the **WESAD** (Wearable Stress and Affect Detection) dataset.

- **Signals Extracted**:
  - **EDA** (Electrodermal Activity): Tonic baseline and phasic sympathetic sweat gland bursts.
  - **ECG / HR**: Heart rate in BPM derived via R-peak peak detection.
  - **Respiration**: Chest expansion cycles.
  - **Skin Temperature**: Peripheral temperature variations.
  - **3-Axis Accelerometer**: Motion intensity and body orientation.
- **Label Mapping**:
  - `0` = **Baseline** (Calm physiological state)
  - `1` = **Stress / Distress** (Elevated sympathetic arousal)
  - `2` = **Amusement** (Dynamic positive affect)

### Downloading Real WESAD Dataset
To download the full official dataset:
```bash
python scripts/download_wesad.py
```
Place any subject folders (`S2/S2.pkl`, `S3/S3.pkl`, etc.) into `data/wesad/`.

---

## How Future Hardware Replaces WESAD

The software uses an abstract `SensorSource` interface (`backend/data/sensor_source.py`):

```python
class SensorSource(ABC):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_next_sample(self) -> Optional[NormalizedPacket]: ...
    def get_current_window(self) -> Optional[np.ndarray]: ...
```

To switch from WESAD to an ESP32 prototype:
1. Connect the ESP32 (running MAX30102 + MPU6050 firmware).
2. The ESP32 transmits JSON packets over Serial UART / TCP:
   ```json
   {
     "timestamp": 1725250000.12,
     "heart_rate": 82.4,
     "eda": 3.45,
     "temperature": 34.2,
     "respiration": 16.5,
     "acceleration": {"x": 0.05, "y": 0.98, "z": 0.12}
   }
   ```
3. Swap `WESADSensorSource` with `ESP32SensorSource` (`backend/data/esp32_adapter.py`).
4. **No other part of the ML, vision, fusion, or UI code needs to be altered.**

---

## Machine Learning & Sequence Modeling

- **Model**: PyTorch `SensorLSTMClassifier`
- **Architecture**:
  - Input: `(Batch, Seq_len=128, Features=5)`
  - Layer 1: LayerNorm + Linear Projection (`hidden_dim=64`)
  - Layer 2: 2-layer LSTM with Dropout (`0.2`)
  - Head: Fully Connected Linear (`64 -> 32 -> 3`) + Softmax
- **Inference Latency**: $\approx 1.5 - 2.5\text{ ms}$ on GPU/CPU.
- **Sensor Distress Score**: Formally calibrated as the softmax probability of the Stress class:
  $$\text{Sensor Distress Score} = P(\text{Stress})$$

---

## Computer Vision Subsystem

- **Model**: YOLOv8n-pose with CUDA acceleration.
- **Keypoint Analysis**: Tracks 17 COCO keypoints and extracts:
  1. **Fall / Lying Ground**: Torso tilt $> 45^\circ$ and bounding box aspect ratio ($H/W < 0.85$).
  2. **Defensive Posture**: Wrists elevated above shoulder level and positioned close to the head/face.
  3. **Rapid Movement / Struggle**: Frame-to-frame keypoint displacement velocity.
- **Visual Distress Score**: Scaled float $[0.0, 1.0]$.
- **Cyber HUD**: Renders neon corner brackets, skeleton vectors, and telemetry HUD directly onto the webcam video stream.

---

## Multimodal Fusion & Decision Logic

### 1. Weighted Decision Base
$$\text{Base Score} = w_{\text{sensor}} \cdot S_{\text{distress}} + w_{\text{vision}} \cdot V_{\text{distress}}$$
*(Defaults: $w_{\text{sensor}} = 0.60, w_{\text{vision}} = 0.40$)*

### 2. Multimodal Consensus & False-Positive Suppression
- **Mutual Escalation**: When $S_{\text{distress}} \ge 0.65$ AND $V_{\text{distress}} \ge 0.65$, confidence is boosted by $\times 1.15$ up to $1.0$.
- **False Alarm Suppression**:
  - High Sensor + Calm Vision (e.g. running or gym exercise): Threat score is capped at $0.65$ (**CAUTION**), preventing single-modality false alarms.
  - High Vision + Low Sensor (e.g. stretching or reaching up): Threat score is capped at $0.60$ (**CAUTION**).

### 3. Alert State Machine & Hysteresis
- `0.00 - 0.39`: **SAFE** (Green `#ccff00`)
- `0.40 - 0.69`: **SUSPICIOUS / CAUTION** (Cyan `#00f0ff`)
- `0.70 - 1.00`: **THREAT / EMERGENCY** (Alert Red `#ff003c`)
- **Temporal Persistence**: Score must remain in the THREAT band for $\ge 2.5\text{s}$ before entering **ALERT ACTIVE** (synthesizes audio chime and logs emergency dispatch).
- **Recovery Buffer**: Must settle below $0.35$ for $\ge 3.0\text{s}$ to reset back to **SAFE**.

---

## Demonstration Scenarios

To ensure deterministic evaluation during the review, 4 demo scenarios are selectable with one click:

| Scenario | Wearable Signal | Vision Posture | Fused Output | Pedagogical Concept |
|---|---|---|---|---|
| **1. Normal Baseline** | Baseline WESAD segment | Calm standing/sitting | **SAFE** ($< 20\%$) | Steady-state baseline operation |
| **2. Stress (Calm Vision)** | High physiological stress | Calm posture | **CAUTION** ($\approx 52\%$) | **False Positive Reduction** (exercise without threat) |
| **3. Multimodal Threat** | High physiological stress | Defensive/struggle posture | **THREAT** ($> 85\%$) | **Multimodal Agreement** trigger |
| **4. False Alarm Filter** | High physiological spike | Normal posture | **CAUTION** ($\approx 50\%$) | Suppression of isolated modality alarms |

Click **⚡ RESUME LIVE STREAM** to return to free continuous streaming.

---

## 📚 Full Documentation

For the complete technical walkthrough — architecture diagrams, the ML stack
explained in plain language (LSTM, YOLOv8-pose keypoints, fusion scoring), the
BLE protocol, hardware wiring, and the dataset-fetching options — see
**[DOCUMENTATION.md](DOCUMENTATION.md)**.

---

## Academic Integrity & Disclaimer

1. **WESAD Dataset**: Physiological data represents laboratory-induced stress and baseline conditions. It is used here as an academic proxy for wearable autonomic response.
2. **Computer Vision**: Detects body geometry (falls, rapid movement, defensive postures) and is described as *visual distress / suspicious activity*, not literal assault classification.
3. **Emergency Dispatch**: The prototype simulates emergency dispatch locally in the audit log; no cellular SMS/GSM network packets are transmitted.

---

## License
Developed for Academic Embedded Systems Capstone Review.
