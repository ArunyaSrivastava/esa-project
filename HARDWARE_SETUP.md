# Physical Hardware Integration Guide
## Personal Safety Monitor: Arduino Uno + MAX30102 + MPU6050 + SSD1306 OLED

This document provides complete instructions for wiring, flashing firmware, and connecting your physical wearable hardware to the Multimodal Threat & Distress Detection System.

---

## 1. Hardware Bill of Materials

| Component | Function | Communication Bus | Default I2C Address |
|---|---|---|---|
| **Arduino Uno (Rev 3 / SMD / Clone)** | Wearable microcontroller & USB serial bridge | USB Serial (115200 baud) | — |
| **MAX30102 Pulse Oximeter & HR Sensor** | Photoplethysmography (PPG), Heart Rate, $\text{SpO}_2$, Finger Detection | I2C (Wire) | `0x57` |
| **MPU6050 6-Axis IMU** | 3D Accelerometer ($a_x, a_y, a_z$) & 3D Gyroscope ($\omega_x, \omega_y, \omega_z$) | I2C (Wire) | `0x68` |
| **SSD1306 0.96" OLED Display (128x64)** | Local wrist HUD showing HR, $\text{SpO}_2$, Motion, and Live Laptop Threat Status | I2C (Wire) | `0x3C` |
| **Breadboard & Jumper Wires** | Prototyping interconnects | — | — |
| **USB Type-A to Type-B Cable** | Powers Arduino and streams serial JSON telemetry to laptop | USB | — |

---

## 2. Complete Wiring Diagram (Shared I2C Bus)

All three modules (MAX30102, MPU6050, SSD1306) communicate over the Arduino Uno's **shared I2C hardware bus** (Pins **A4** and **A5**).

```
           +---------------------------------------------+
           |                 ARDUINO UNO                 |
           |                                             |
           |   5V   GND        A4 (SDA)      A5 (SCL)    |
           +----+----+------------+-------------+--------+
                |    |            |             |
                |    |            |             |
  +-------------+----+------------+-------------+--------+
  |             |    |            |             |        |
  |  +----------+----+------------+-------------+-----+  |
  |  | MAX30102 (PPG / HR / SpO2)                      |  |
  |  | VCC: 5V/3.3V  GND: GND   SDA: A4       SCL: A5  |  |
  |  +------------------------------------------------+  |
  |                                                      |
  |  +------------------------------------------------+  |
  |  | MPU6050 (6-Axis IMU)                            |  |
  |  | VCC: 5V       GND: GND   SDA: A4       SCL: A5  |  |
  |  +------------------------------------------------+  |
  |                                                      |
  |  +------------------------------------------------+  |
  |  | SSD1306 OLED (128x64 Display)                   |  |
  |  | VCC: 5V       GND: GND   SDA: A4       SCL: A5  |  |
  +--+------------------------------------------------+--+
```

### Pin-to-Pin Connection Table

| Arduino Uno Pin | MAX30102 Pin | MPU6050 Pin | SSD1306 OLED Pin |
|---|---|---|---|
| **5V** (or 3.3V*) | `VIN` or `VCC` | `VCC` | `VCC` |
| **GND** | `GND` | `GND` | `GND` |
| **Pin A4 (SDA)** | `SDA` | `SDA` | `SDA` |
| **Pin A5 (SCL)** | `SCL` | `SCL` | `SCL` |

> [!TIP]
> * **Power Note**: Most standard breakout modules (e.g. GY-521 for MPU6050, 4-pin I2C OLED, and red MAX30102 boards) include an on-board 3.3V LDO regulator and can be powered directly from Arduino **5V**.
> * **I2C Pull-Up Resistors**: The sensor breakouts have integrated pull-up resistors on SDA/SCL, so no external resistors are necessary.

---

## 3. Arduino IDE Setup & Firmware Upload

### Step 1: Install Required Arduino Libraries
Open the **Arduino IDE** and go to **Tools $\rightarrow$ Manage Libraries...** (or press `Ctrl+Shift+I`). Search for and install:
1. **Adafruit SSD1306** (by Adafruit) — Install with all dependencies (including *Adafruit GFX Library*).
2. **SparkFun MAX3010x Pulse and Proximity Sensor Library** (by SparkFun Electronics).

### Step 2: Open and Upload the Firmware Sketch
1. In Arduino IDE, open:
   ```
   firmware/arduino_wearable/arduino_wearable.ino
   ```
2. In the **Tools** menu:
   - **Board**: Select `Arduino Uno`.
   - **Port**: Select the COM port corresponding to your plugged-in Arduino (e.g. `COM3` or `COM4`).
3. Click **Upload** (`Ctrl+U`).
4. Once uploaded:
   - The SSD1306 OLED will display:
     ```
     PERSONAL SAFETY
     MONITOR - UNO
     Init Sensors...
     ```
   - Followed by the live wrist HUD showing Heart Rate, $\text{SpO}_2$, Acceleration ($g$), and Threat Status.

---

## 4. Connecting to the System Dashboard

1. Launch the system backend on your laptop:
   ```powershell
   python run.py
   ```
2. Open your browser at **`http://localhost:8000`**.
3. In the top navbar or the **Sensor Source** panel:
   - Select your Arduino COM port (e.g., `COM3`).
   - Click **`🔌 Connect Arduino`**.
4. The dashboard will automatically switch to **`ARDUINO LIVE`** mode:
   - **Wearable Sensors card**: Displays your real-time **Heart Rate (BPM)**, **$\text{SpO}_2$ (%)**, **Motion ($g$)**, and **Gyro Rotation ($^\circ/\text{s}$)**.
   - **Zero Fabricated Signals**: Shows strictly authentic physical data.
   - **OLED Mirroring**: The SSD1306 OLED display on your Arduino will mirror the laptop's live multimodal threat verdict (`[ SAFE ]`, `[ CAUTION ]`, or `[ THREAT! ]`) with a progress meter in real time!
5. To switch back to the WESAD recording benchmark at any time, simply click **`⚡ WESAD Replay`**.

---

## 5. Troubleshooting & Verification

| Issue | Root Cause | Solution |
|---|---|---|
| **OLED stays blank on boot** | I2C wiring loose or address mismatch | Verify A4 (SDA) and A5 (SCL) connections. Most SSD1306 displays use address `0x3C`. |
| **Heart Rate displays `-- BPM`** | No finger placed or inadequate pressure | Gently rest your index finger flat on the MAX30102 glass sensor. Avoid pressing too hard. |
| **COM Port not appearing in UI** | USB driver missing or cable is charge-only | Ensure your USB cable supports data transfer. Check Windows Device Manager under *Ports (COM & LPT)*. |
| **Serial collision error** | Arduino IDE Serial Monitor is currently open | Close the Arduino IDE Serial Monitor window before connecting via the web dashboard. |
