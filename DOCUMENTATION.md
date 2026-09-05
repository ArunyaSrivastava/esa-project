# Multimodal Threat & Distress Detection System — Technical Documentation

*Written to make the system easy to explain to reviewers, professors, and
stakeholders. Terms are introduced in plain language, then given the exact
technical name used in the codebase.*

> If you only read one section, read **Section 1** (What it does, in plain
> terms) and **Section 4** (The ML stack). Sections 5–6 cover the right and
> hardware side in depth.

---

## Table of Contents
1. [What the system does (plain terms)](#1-what-the-system-does-in-plain-terms)
2. [High-level architecture](#2-high-level-architecture)
3. [The two sensing modalities](#3-the-two-sensing-modalities)
4. [The machine-learning stack](#4-the-machine-learning-stack)
5. [Multimodal fusion & decision logic](#5-multimodal-fusion--decision-logic)
6. [Hardware & wearable implementation](#6-hardware--wearable-implementation)
7. [Project layout](#7-project-layout)
8. [How to run it](#8-how-to-run-it)
9. [Obtaining training data (internet options)](#9-obtaining-training-data-internet-options)
10. [Glossary / jargon buster](#10-glossary--jargon-buster)

---

## 1. What the system does (in plain terms)

This is a **safety-monitoring prototype**. It watches a person using **two
kinds of information at once** and raises an alarm only when *both* point to a
real problem — which sharply reduces false alarms.

**Information source A — the body (wearable):** a wearable sensor measures
physiological signals (heart rate, oxygen saturation, motion). When a person is
under physiological stress their heart rate rises and their motion may become
erratic.

**Information source B — the camera (vision):** a laptop camera feeds a
pose-estimation model that detects *posture* — whether someone has fallen, is
raising their arms defensively, or is moving violently.

**The decision:** both streams are combined ("fused") into a single **threat
score** 0.0 → 1.0. The system maps that score to a state:
- **SAFE** (green) — no concern
- **CAUTION / SUSPICIOUS** (cyan) — something elevated but not alarming
- **THREAT / EMERGENCY** (red) — escalate

The key design principle: **one signal alone rarely triggers a full alert.**
High heart rate *plus* a defensive posture → threat. High heart rate *while*
calmly walking → caution, not a false emergency. This "multimodal agreement"
idea is the intellectual core you can emphasize in a review.

> Smooth term to use: the system performs **cross-modal consensus-based threat
> assessment with temporal persistence and hysteresis** (more on that in §5).

---

## 2. High-level architecture

```
   WEARABLE (physiology)                     CAMERA (vision)
   MAX30102 PPG  ─┐                           ┌─ YOLOv8-pose → 17 keypoints
   MPU6050 IMU   ─┼─ NormalizedPacket ─┐      ├─ fall / defensive / struggle
   (or WESAD      │                     │      └─ motion velocity
    replay file)  │                     │
                  ▼                     ▼
        [ Sensor ML / heuristics ]   [ Visual distress score ]
                  │                     │
                  └──────────┬──────────┘
                             ▼
                 MULTIMODAL FUSION ENGINE
                  (weighted score + agreement)
                             ▼
                 ALERT STATE MACHINE
                  (SAFE → CAUTION → THREAT → ALERT)
                             ▼
                 WebSocket → Browser Dashboard (real-time)
```

Key architectural win: the **SensorSource abstraction** means the system does
not care whether the body data comes from a saved dataset file (WESAD), from a
USB-connected Arduino, or from a wireless ESP32 over Bluetooth — they all emit
the same `NormalizedPacket`. The ML, fusion, and UI layers are unchanged.

---

## 3. The two sensing modalities

### 3.1 Physiological (wearable) signals

| Signal | Source (real HW) | What it tells us |
|---|---|---|
| **PPG / Heart rate** | MAX30102 (reflective pulse oximeter) | Beats-per-minute; stress raises HR |
| **SpO₂** (blood oxygen) | MAX30102 | Oxygen saturation; drops in some emergencies |
| **3-axis acceleration** | MPU6050 IMU | Impact/g forces; fall detection |
| **3-axis gyroscope** | MPU6050 IMU | Rotational velocity; struggle detection |

*PPG* (photoplethysmography) is the optical technique: an LED shines into the
skin and a photodiode measures the changing blood volume with each heartbeat.

The demo also replays the **WESAD dataset**, a public research dataset, to
stand in for the wearable (see §9).

### 3.2 Vision signals

- **YOLOv8-pose** detects a person and returns **17 body keypoints**
  (nose, shoulders, elbows, wrists, hips, knees, ankles).
- From those keypoints we compute three risk features:
  1. **Fall / lying** — torso tilt and width/height ratio.
  2. **Defensive posture** — wrists raised near the head/face.
---

## 4. The machine-learning stack

There are **four decision components**. It is important to state clearly which
are "deep learning" and which are rule-based:

| Component | Type | File | Output |
|---|---|---|---|
| **Sensor LSTM** | Deep learning (RNN) | `backend/ml/` | P(Stress) 0→1 |
| **YOLOv8-pose** | Deep learning (CNN) | `backend/vision/` | Person + 17 keypoints |
| **Wearable inference** | Rule-based heuristics | `backend/ml/hardware_inference.py` | Distress score 0→1 |
| **Fusion engine** | Weighted consensus + state machine | `backend/fusion/` | Threat score 0→1 |

### 4.1 The LSTM sensor classifier
An **LSTM (Long Short-Term Memory)** is a **recurrent neural network** — a
network with internal "memory", well suited to time-series. It reads a sliding
window of physiological signals and classifies the person's state:

- **Input:** sliding windows of 5 channels — EDA, ECG/HR, respiration, skin
  temperature, acceleration magnitude (from WESAD @ ~32 Hz effective rate).
- **Architecture:** 2 stacked LSTM layers (hidden size 64) → dropout →
  fully-connected head → **3-class softmax** (Baseline / Stress / Amusement).
- **Sensor Distress Score** = the softmax probability of the **Stress** class.
- Training runs offline (`scripts/train_model.py`); the app only *infers*.

> Plain terms: *"The network reads the last few seconds of the body's signals
> and answers: is this person calm, stressed, or amused? High stress
> probability = evidence of physiological distress."*

### 4.2 YOLOv8-pose (vision)
**YOLO (You Only Look Once)** is a single-pass CNN detector. The **pose**
variant also regresses **17 COCO keypoints** (nose, shoulders, elbows, wrists,
hips, knees, ankles). We run `yolov8n-pose` (the "nano" variant) for real-time
speed, then derive three risk features from the keypoint geometry:
1. **Fall / lying** — torso tilt > 45°, bounding-box height/width < 0.85.
2. **Defensive posture** — wrists elevated near head/face.
3. **Struggle** — frame-to-frame keypoint velocity above threshold.

### 4.3 Wearable pipeline (real hardware)
For real MAX30102 + MPU6050 data we use **explainable thresholds** rather than
the LSTM (the LSTM was trained on chest-ECG WESAD features, not our wearable):

| Signal | Threshold | Meaning |
|---|---|---|
| Heart rate | > 125 bpm | Tachycardia → strong distress evidence |
| SpO₂ | < 90 % | Hypoxia penalty |
| \|accel\| | > 2.6 g | Impact / possible fall |
| Gyro rate | > ~140°/s | Violent struggle |

These combine additively into a `[0,1]` score. This keeps the live-hardware
path honest, interpretable, and defensible in a review, while the LSTM remains
the "sequence model" showcase on dataset replay.

---

## 5. Multimodal fusion & decision logic

### 5.1 Weighted base
```
Final score = w_sensor × sensor_distress + w_vision × visual_distress
Defaults:   w_sensor = 0.60,        w_vision  = 0.40
```
Weights are adjustable live from the dashboard sliders.

### 5.2 Cross-modal agreement (the false-alarm guard)
- **Both high** (sensor ≥ 0.65 AND vision ≥ 0.60) → confidence boosted ×1.15
  → **THREAT** path.
- **Sensor high, vision calm** (e.g. exercise) → score **capped ≈ 0.65/0.50**
  → CAUTION, never emergency.
- **Vision high, sensor calm** (e.g. stretching) → score **capped ≈ 0.60/0.46**
  → CAUTION.

One modality alone can raise **caution**, but never a full **emergency** — that
requires cross-modal agreement. This is the core intellectual claim of the
project and the headline of any review.

### 5.3 Temporal persistence & hysteresis
The **AlertStateMachine** (`backend/fusion/state_machine.py`) filters spikes:
- Score must stay ≥ 0.70 for **≥ 2.5 s** before entering ALERT ACTIVE
  (chime + audit-log dispatch).
- Must settle < 0.35 for **≥ 3.0 s** to reset to SAFE.

> Quote-ready: *the system applies temporal persistence and hysteresis so a
> momentary blip cannot trigger an emergency.*

### 5.4 One-click demo scenarios
| Scenario | Sensor | Vision | Result | Concept shown |
|---|---|---|---|---|
| Normal Baseline | calm | calm | SAFE (<20%) | steady state |
| Stress, Calm Vision | high | calm | CAUTION ≈52% | false-positive reduction |
| Multimodal Threat | high | defensive | THREAT >85% | consensus trigger |
| False Alarm Filter | spike | calm | CAUTION ≈50% | suppression |

  3. **Rapid movement / struggle** — frame-to-frame keypoint velocity.
---

## 6. Hardware & wearable implementation

### 6.1 Tier-1 wireless wearable (the build)
| Part | Role | Notes / cost |
|---|---|---|
| **ESP32-C3 SuperMini** | MCU + BLE 5 radio | ~$3; tiny USB-C board |
| **MAX30102** | PPG → HR + SpO₂ | Reflective pulse oximetry, I²C 0x57 |
| **MPU6050** | 3-axis accel + 3-axis gyro | I²C 0x68 |
| **TP4056** module | LiPo charge + protection | ~$2.5 |
| 300 mAh LiPo | Power | ~$2.5; hours of BLE streaming |
| Mini switch + strap | Enclosure | ~$2 |

### 6.2 Wiring (ESP32-C3, shared I²C bus)
| ESP32-C3 pin | Connected to |
|---|---|
| 3V3 | MAX30102 VIN, MPU6050 VCC |
| GND | Both GND + TP4056 GND |
| GPIO 8 / GPIO 9 | I²C SDA / SCL (ESP32-C3 defaults) |
| TP4056 BAT+ → 3V3 rail | Battery power |

> Anticipated question — *why measure motion at all?* Because PPG suffers
> **motion artifacts**: when the hand moves, the optical pulse signal is
> contaminated. Reading the IMU at the same time lets software tell a real
> heart-rate rise apart from sensor movement — which is exactly why fusion
> weighs both channels together.

### 6.3 BLE protocol (firmware ↔ laptop)
- Device advertises as **`ESA-WEAR-01`**.
- Service UUID: `8d54b3a0-0000-0000-0000-ffff0000c3ec`
- Notify characteristic: `8d54b3a0-0001-0000-0000-ffff0000c3ec`
- Payload: one JSON object per notification at **15 Hz**:
```json
{"hr": 78, "spo2": 98.0, "ax": 0.02, "ay": 0.99, "az": 0.10,
 "gx": 1.1, "gy": -0.4, "gz": 0.8, "mag": 1.00, "finger": 1}
```
- `finger` = MAX30102 contact-detect bit. When 0, the adapter marks the packet
  low-quality and fusion ignores HR/SpO₂ for that instant.
- Laptop side: `backend/data/ble_adapter.py` (asyncio via **bleak**) runs in a
  background thread and emits the **same `NormalizedPacket`** as the USB serial
  path — a drop-in replacement; ML, fusion, and UI are untouched.

### 6.4 Power note (why a 5-minute demo needs no optimisation)
At 15 Hz the BLE radio is duty-cycled briefly every ~66 ms; comparable builds
draw ~25–40 mA average, so a 300 mAh cell runs **>6 hours** — far beyond demo
needs. No sleep scheduling is required for the review.

---

## 7. Project layout

```
esa-project/
├── backend/
│   ├── main.py               # FastAPI app + WebSocket + source factory
│   ├── config.py             # Wearable source/BLE config, thresholds
│   ├── data/
│   │   ├── ble_adapter.py    # ESP32 BLE → NormalizedPacket (bleak)
│   │   ├── arduino_adapter.py# USB serial path (Arduino Uno)
│   │   ├── replay_engine.py  # WESAD replay with play/pause/speed/seek
│   │   └── preprocessing.py  # resample + standardise pipeline
│   ├── ml/
│   │   ├── model.py          # 2-layer LSTM definition
│   │   ├── inference.py      # real-time inference wrapper
│   │   ├── train.py          # offline training
│   │   └── hardware_inference.py  # rule-based wearable scoring
│   ├── fusion/
│   │   ├── fusion_engine.py  # weighted score + cross-modal agreement
│   │   └── state_machine.py  # SAFE/CAUTION/THREAT + hysteresis
│   └── vision/
│       ├── detector.py       # YOLOv8-pose wrapper
│       └── camera.py         # macOS AVFoundation-safe capture
├── firmware/
│   ├── arduino_wearable/     # USB-serial wearable (Tier 0)
│   └── esp32_wearable/       # BLE wearable (Tier 1) ← the demo build
├── frontend/
│   ├── index.html            # 8 tagged dashboard modules
│   ├── js/app.js             # module toggles + live UI
│   └── css/style.css         # dark-neon dashboard theme
├── scripts/
│   ├── fetch_datasets.py     # internet dataset puller (uci / sciebo)
│   ├── download_wesad.py     # original WESAD zip downloader
│   ├── preprocess_wesad.py   # sliding windows + global standardiser
│   ├── train_model.py        # LSTM training entrypoint
│   └── test_ble_adapter.py   # BLE payload parse unit test
├── models/                   # trained checkpoint + scaler params
├── run.py / run_macos.sh     # launchers
└── DOCUMENTATION.md          # this document
```

---

## 8. How to run it

```bash
cd esa-project
./run_macos.sh          # creates .venv, installs deps, launches server
# → open http://localhost:8000
```

**Live wearable demo checklist:**
1. Power on the ESP32 wearable (slide switch).
2. Allow Bluetooth permission for Terminal/iTerm when macOS prompts.
3. Dashboard → Modules → ☑ **Wearable** (status flips to `LIVE` on connect).
4. Wear it; raise heart rate (stairs / fast speech) and raise arms toward the
   face — fusion should cross 0.70 → THREAT after ~2.5 s of persistence.
5. Hide all technical modules for a clean view; the fusion card and camera
   carry the whole story.

---

## 9. Obtaining training data (internet options)

The model is trained on **WESAD** — a public dataset of 15 subjects wearing
chest + wrist sensors during baseline, stress, and amusement conditions.
License note: WESAD's formal terms require a signed request via the University
of Siegen / PhysioNet; the UCI mirror is the practical programmatic route.

```bash
# Route A (default, robust): UCI Machine Learning Repository API
python scripts/fetch_datasets.py --source uci

# Route A, smaller first download (3 subjects only)
python scripts/fetch_datasets.py --source uci --limit-subjects S2 S3 S4

# Route B: original University of Siegen full archive (~2 GB zip)
python scripts/fetch_datasets.py --source sciebo

# Then:
python scripts/preprocess_wesad.py     # sliding windows + standardiser
python scripts/train_model.py          # trains LSTM, saves to models/
```

If no dataset is present, the backend transparently falls back to the bundled
**synthetic** WESAD generator so the full pipeline (train → infer → dashboard)
still runs end-to-end for a review.

---

## 10. Glossary / jargon buster

| Term | Plain meaning |
|---|---|
| **PPG** | Optical pulse sensing — LED light into skin, photodiode sees blood-volume pulses → heart rate |
| **SpO₂** | Blood-oxygen percentage measured from red/IR light absorption |
| **IMU** | Inertial Measurement Unit — accelerometer + gyroscope chip |
| **LSTM** | Recurrent neural network with memory; good at ordered time-series |
| **RNN** | Neural network that processes sequences step-by-step, carrying state |
| **YOLOv8-pose** | One-pass CNN that finds people and their 17 body joints |
| **Keypoint** | A named body-joint coordinate (x, y, confidence) |
| **Softmax probability** | Network's normalised confidence across classes (sums to 1) |
| **Inference** | Running a trained model on live data (vs. training) |
| **Normalised / standardised** | Rescaled to comparable ranges (zero mean, unit variance) |
| **Sliding window** | Take overlapping chunks of a signal, classify each chunk |
| **Fusion** | Combining multiple independent evidence sources into one decision |
| **Hysteresis** | Requiring conditions to persist/settle before switching state |
| **BLE notify** | Bluetooth mode where the device pushes small packets continuously |
| **I²C** | Two-wire bus (SDA/SCL) letting the MCU talk to both sensor chips |
| **Motion artifact** | Noise in an optical pulse signal caused by movement |
| **TP4056** | Single-cell LiPo charging + protection module |
| **ESP32-C3** | Low-cost RISC-V MCU with WiFi + Bluetooth LE on-chip |
| **False positive** | Alarm raised when there is no real problem |
| **Multimodal** | Using >1 sensing modality (body + vision) for the same decision |

---

## 11. Academic integrity & limitations
1. **WESAD** reflects *laboratory-induced* stress; it is an academic proxy for
   autonomic response, not field data.
2. **Vision** detects body geometry (falls, rapid movement, defensive posture),
   described as *visual distress / suspicious activity*, not literal assault
   classification.
3. **Emergency dispatch** is simulated locally in the audit log; no SMS/GSM is
   transmitted.
4. The demo is a **prototype** for research review, not a certified safety
   device.