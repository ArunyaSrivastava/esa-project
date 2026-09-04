/*
 * ==============================================================================
 * PERSONAL SAFETY MONITOR - ARDUINO UNO WEARABLE FIRMWARE
 * ==============================================================================
 * Hardware:
 *   - Arduino Uno (ATmega328P)
 *   - MAX30102 Optical Pulse Oximeter & Heart Rate Sensor (I2C: 0x57)
 *   - MPU6050 6-Axis Motion Tracking IMU (I2C: 0x68)
 *   - SSD1306 0.96" OLED Display 128x64 (I2C: 0x3C)
 *
 * Wiring (Shared I2C Bus):
 *   - All SDA pins -> Arduino Uno Pin A4
 *   - All SCL pins -> Arduino Uno Pin A5
 *   - VCC -> 5V (or 3.3V for MAX30102 if module lacks on-board regulator)
 *   - GND -> Arduino GND
 *
 * Serial Protocol:
 *   - Baud Rate: 115200 bps
 *   - Upstream (Arduino -> Laptop):
 *     {"hr":76.2,"spo2":98.0,"ax":0.02,"ay":0.99,"az":0.10,"gx":1.1,"gy":-0.4,"gz":0.8,"mag":1.00,"finger":1}
 *   - Downstream (Laptop -> Arduino OLED feedback):
 *     {"threat":15,"state":"SAFE"}
 * ==============================================================================
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "MAX30105.h"
#include "heartRate.h"

// OLED Display Configuration
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// MAX30102 Sensor
MAX30105 particleSensor;

// MPU6050 I2C Register Addresses
const int MPU_ADDR = 0x68;

// Sampling & Heart Rate State
const byte RATE_SIZE = 4;
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;
float beatsPerMinute = 0;
int beatAvg = 0;
float estimatedSpO2 = 98.0;

// Motion Variables (g & deg/s)
float ax = 0, ay = 0, az = 1.0;
float gx = 0, gy = 0, gz = 0;
float acc_mag = 1.0;
bool fingerDetected = false;

// Laptop Feedback State (Mirroring Multimodal Fusion Decision)
int laptopThreatScore = 0;
String laptopState = "INITIALIZING";
unsigned long lastSerialSend = 0;
unsigned long lastOledRefresh = 0;
const unsigned long SEND_INTERVAL_MS = 66;   // ~15 Hz serial telemetry rate
const unsigned long OLED_INTERVAL_MS = 150;  // ~6.6 Hz OLED refresh

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000); // 400kHz Fast I2C mode

  // 1. Initialize OLED Display
  if (display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(10, 15);
    display.println("PERSONAL SAFETY");
    display.setCursor(10, 30);
    display.println("MONITOR - UNO");
    display.setCursor(10, 48);
    display.println("Init Sensors...");
    display.display();
  }

  // 2. Initialize MPU6050
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); // PWR_MGMT_1 register
  Wire.write(0);    // Wake up MPU6050
  Wire.endTransmission(true);

  // 3. Initialize MAX30102
  if (particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    // Configure sensor for Heart Rate & SpO2 mode
    byte ledBrightness = 60; // 0=Off to 255=50mA
    byte sampleAverage = 4;  // 1, 2, 4, 8, 16, 32
    byte ledMode = 2;        // 2 = Red + IR (SpO2 mode)
    byte sampleRate = 100;   // 50, 100, 200, 400, 800, 1000, 1600, 3200
    int pulseWidth = 411;    // 69, 118, 215, 411
    int adcRange = 4096;     // 2048, 4096, 8192, 16384

    particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
    particleSensor.setPulseAmplitudeRed(0x0A); // Low power red LED
    particleSensor.setPulseAmplitudeGreen(0);  // Turn off green
  }

  delay(500);
}

void loop() {
  unsigned long now = millis();

  // 1. Read MAX30102 PPG & Compute Heart Rate
  long irValue = particleSensor.getIR();
  long redValue = particleSensor.getRed();

  if (irValue > 50000) {
    fingerDetected = true;

    // Detect optical peak (heartbeat)
    if (checkForBeat(irValue) == true) {
      long delta = now - lastBeat;
      lastBeat = now;

      if (delta > 300 && delta < 2000) {
        beatsPerMinute = 60 / (delta / 1000.0);
        if (beatsPerMinute > 40 && beatsPerMinute < 220) {
          rates[rateSpot++] = (byte)beatsPerMinute;
          rateSpot %= RATE_SIZE;

          long sum = 0;
          for (byte x = 0; x < RATE_SIZE; x++) sum += rates[x];
          beatAvg = sum / RATE_SIZE;
        }
      }
    }

    // Standard PPG R-ratio approximation for SpO2
    if (redValue > 0 && irValue > 0) {
      float ratio = ((float)redValue / (float)irValue);
      float spo2Calc = 110.0 - 15.0 * ratio;
      if (spo2Calc >= 80.0 && spo2Calc <= 100.0) {
        estimatedSpO2 = 0.9 * estimatedSpO2 + 0.1 * spo2Calc;
      }
    }
  } else {
    fingerDetected = false;
    beatAvg = 0;
  }

  // 2. Read MPU6050 6-Axis Acceleration & Gyroscope
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); // Starting register for accelerometer data
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, true);

  if (Wire.available() >= 14) {
    int16_t raw_ax = (Wire.read() << 8) | Wire.read();
    int16_t raw_ay = (Wire.read() << 8) | Wire.read();
    int16_t raw_az = (Wire.read() << 8) | Wire.read();
    int16_t raw_temp = (Wire.read() << 8) | Wire.read(); // Unused
    int16_t raw_gx = (Wire.read() << 8) | Wire.read();
    int16_t raw_gy = (Wire.read() << 8) | Wire.read();
    int16_t raw_gz = (Wire.read() << 8) | Wire.read();

    // Scale to standard units (+-2g range: 16384 LSB/g, +-250 deg/s: 131 LSB/deg/s)
    ax = (float)raw_ax / 16384.0;
    ay = (float)raw_ay / 16384.0;
    az = (float)raw_az / 16384.0;
    acc_mag = sqrt(ax * ax + ay * ay + az * az);

    gx = (float)raw_gx / 131.0;
    gy = (float)raw_gy / 131.0;
    gz = (float)raw_gz / 131.0;
  }

  // 3. Read Downstream Feedback from Laptop (Non-blocking)
  if (Serial.available() > 0) {
    String inputLine = Serial.readStringUntil('\n');
    parseLaptopFeedback(inputLine);
  }

  // 4. Send Upstream JSON Telemetry over Serial to Laptop (~15 Hz)
  if (now - lastSerialSend >= SEND_INTERVAL_MS) {
    lastSerialSend = now;

    Serial.print("{\"hr\":");
    Serial.print(beatAvg);
    Serial.print(",\"spo2\":");
    Serial.print(estimatedSpO2, 1);
    Serial.print(",\"ax\":");
    Serial.print(ax, 2);
    Serial.print(",\"ay\":");
    Serial.print(ay, 2);
    Serial.print(",\"az\":");
    Serial.print(az, 2);
    Serial.print(",\"gx\":");
    Serial.print(gx, 1);
    Serial.print(",\"gy\":");
    Serial.print(gy, 1);
    Serial.print(",\"gz\":");
    Serial.print(gz, 1);
    Serial.print(",\"mag\":");
    Serial.print(acc_mag, 2);
    Serial.print(",\"finger\":");
    Serial.print(fingerDetected ? 1 : 0);
    Serial.println("}");
  }

  // 5. Update SSD1306 OLED Display (~6.6 Hz)
  if (now - lastOledRefresh >= OLED_INTERVAL_MS) {
    lastOledRefresh = now;
    renderOLED();
  }
}

// Parse Laptop Feedback JSON: {"threat":15,"state":"SAFE"}
void parseLaptopFeedback(String line) {
  line.trim();
  if (line.startsWith("{") && line.endsWith("}")) {
    int threatIdx = line.indexOf("\"threat\":");
    if (threatIdx != -1) {
      int commaIdx = line.indexOf(",", threatIdx);
      if (commaIdx == -1) commaIdx = line.indexOf("}", threatIdx);
      laptopThreatScore = line.substring(threatIdx + 9, commaIdx).toInt();
    }

    int stateIdx = line.indexOf("\"state\":\"");
    if (stateIdx != -1) {
      int endQuote = line.indexOf("\"", stateIdx + 9);
      if (endQuote != -1) {
        laptopState = line.substring(stateIdx + 9, endQuote);
      }
    }
  }
}

// Render Clean Local HUD on OLED
void renderOLED() {
  display.clearDisplay();

  // Top Status Bar
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("SAFETY: ");
  if (laptopThreatScore >= 70 || laptopState.indexOf("THREAT") != -1) {
    display.print("[ THREAT! ]");
  } else if (laptopThreatScore >= 40 || laptopState.indexOf("CAUTION") != -1) {
    display.print("[ CAUTION ]");
  } else {
    display.print("[ SAFE ]");
  }
  display.drawLine(0, 10, 128, 10, SSD1306_WHITE);

  // Line 1: Heart Rate & SpO2
  display.setCursor(0, 14);
  if (fingerDetected) {
    display.print("HR: ");
    display.print(beatAvg);
    display.print(" BPM");

    display.setCursor(72, 14);
    display.print("O2:");
    display.print((int)estimatedSpO2);
    display.print("%");
  } else {
    display.print("HR: -- (Place Finger)");
  }

  // Line 2: Acceleration & Gyroscope
  display.setCursor(0, 28);
  display.print("Acc: ");
  display.print(acc_mag, 2);
  display.print("g");

  display.setCursor(72, 28);
  display.print("Gyr: ");
  display.print((int)sqrt(gx * gx + gy * gy + gz * gz));

  // Line 3: Threat Score
  display.setCursor(0, 42);
  display.print("Threat: ");
  display.print(laptopThreatScore);
  display.print("%");

  // Bottom Progress Bar
  int barWidth = map(constrain(laptopThreatScore, 0, 100), 0, 100, 0, 128);
  display.drawRect(0, 54, 128, 8, SSD1306_WHITE);
  display.fillRect(0, 54, barWidth, 8, SSD1306_WHITE);

  display.display();
}
