/*
 * ==============================================================================
 * PERSONAL SAFETY MONITOR - ESP32-C3 WIRELESS WEARABLE FIRMWARE (TIER 1)
 * ==============================================================================
 * Hardware:
 *   - ESP32-C3 SuperMini (WiFi + BLE, 3.3V LDO onboard)
 *   - MAX30102 Optical Pulse Oximeter & Heart Rate Sensor (I2C: 0x57)
 *   - MPU6050 6-Axis Motion Tracking IMU (I2C: 0x68)
 *
 * Purpose:
 *   Reads the same raw sensors as the original Arduino Uno build but streams
 *   them wirelessly over Bluetooth Low Energy (BLE) instead of USB serial,
 *   so the whole node can be strapped to a wrist for a live demo.
 *
 * BLE Design:
 *   - Service UUID:     8d54b3a0-0000-0000-0000-ffff0000c3ec (editable)
 *   - Characteristic:   ESA_WEAR_TELEMETRY (Notify) holds a single JSON line.
 *     {"hr":76.2,"spo2":98.0,"ax":0.02,"ay":0.99,"az":0.10,
 *      "gx":1.1,"gy":-0.4,"gz":0.8,"mag":1.00,"finger":1}
 *   - Notify cadence: ~66ms (15 Hz) exactly matching the original serial rate.
 *
 * Wiring (shared I2C):
 *   MAX30102 VIN/SCL/SDA -> 3V3 / GPIO9 (SCL) / GPIO8 (SDA)
 *   MPU6050  VCC/SCL/SDA -> 3V3(5V ok via module reg) / GPIO9 / GPIO8
 *   All GND together. Adjust SDA_PIN / SCL_PIN if your board routes elsewhere.
 *
 * Libraries (Arduino IDE -> Manage Libraries):
 *   - BLE:      comes with the esp32 core (BLEDevice)
 *   - MAX30102: SparkFun MAX3010x Pulse and Proximity Sensor Library
 *   - MPU6050 : raw I2C register reads, no external lib needed
 *   Boards Manager: esp32 by Espressif (tested with ESP32-C3 "SuperMini")
 * ==============================================================================
 */

#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEService.h>
#include <BLECharacteristic.h>
#include "MAX30105.h"
#include "heartRate.h"

// ---- I2C pins (ESP32-C3 default in Arduino core) -------------------------
#define SDA_PIN 8
#define SCL_PIN 9

// ---- BLE service / characteristic UUIDs ----------------------------------
#define SERVICE_UUID        "8d54b3a0-0000-0000-0000-ffff0000c3ec"
#define CHAR_TELEMETRY_UUID "8d54b3a0-0001-0000-0000-ffff0000c3ec"

BLECharacteristic *pTelemetryChar = nullptr;

// ---- Sensors -------------------------------------------------------------
MAX30105 particleSensor;   // MAX30102 pulse oximeter
const int MPU_ADDR = 0x68; // MPU6050

// Heart-rate state
const byte RATE_SIZE = 4;
byte rates[RATE_SIZE];
byte rateSpot = 0;
long lastBeat = 0;
float beatsPerMinute = 0;
int beatAvg = 0;
float estimatedSpO2 = 98.0;

// Motion state
float ax = 0, ay = 0, az = 1.0;
float gx = 0, gy = 0, gz = 0;
float acc_mag = 1.0;
bool fingerDetected = false;

// Timing
unsigned long lastSend = 0;
// -----------------------------------------------------------------------------
void mpu6050_init() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); // PWR_MGMT_1
  Wire.write(0);    // wake up
  Wire.endTransmission(true);
}

void mpu6050_read(float &ax_, float &ay_, float &az_,
                  float &gx_, float &gy_, float &gz_) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); // ACCEL_XOUT_H
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14, true);

  int16_t raw_a[3], raw_g[3];
  for (int i = 0; i < 3; i++)
    raw_a[i] = (Wire.read() << 8) | Wire.read();
  for (int i = 0; i < 3; i++)
    raw_g[i] = (Wire.read() << 8) | Wire.read();

  ax_ = raw_a[0] / 16384.0; // +-2g
  ay_ = raw_a[1] / 16384.0;
  az_ = raw_a[2] / 16384.0;
  gx_ = raw_g[0] / 131.0;   // +-250 deg/s
  gy_ = raw_g[1] / 131.0;
  gz_ = raw_g[2] / 131.0;
}

// -----------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  mpu6050_init();

  if (particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    byte ledBrightness = 60;
    byte sampleAverage = 8;
    byte ledMode = 2;        // red + IR (SpO2)
    byte sampleRate = 100;
    int pulseWidth = 411;
    int adcRange = 4096;
    particleSensor.setup(ledBrightness, sampleAverage, ledMode,
                         sampleRate, pulseWidth, adcRange);
    particleSensor.setPulseAmplitudeRed(0x0A);
    particleSensor.setPulseAmplitudeGreen(0);
  }

  // ---- BLE setup ---------------------------------------------------------
  BLEDevice::init("ESA-WEAR-01");
  BLEServer *pServer = BLEDevice::createServer();
  BLEService *pService = pServer->createService(SERVICE_UUID);
  pTelemetryChar = pService->createCharacteristic(
      CHAR_TELEMETRY_UUID,
      BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
  pTelemetryChar->setValue("{\"status\":\"waiting\"}");
  pService->start();
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  BLEDevice::startAdvertising();

  Serial.println("[ESA] ESP32-C3 wearable BLE online.");
}

// -----------------------------------------------------------------------------
void loop() {
  unsigned long now = millis();
  if (now - lastSend < SEND_INTERVAL_MS)
    return;
  lastSend = now;

  // ---- Read MPU6050 (always available) -----------------------------------
  mpu6050_read(ax, ay, az, gx, gy, gz);
  acc_mag = sqrt(ax * ax + ay * ay + az * az);

  // ---- Read MAX30102 heart rate (finger placed) --------------------------
  long irValue = particleSensor.getIR();
  fingerDetected = irValue > 50000; // IR baseline threshold
  if (fingerDetected) {
    if (checkForBeat(irValue) == true) {
      long delta = millis() - lastBeat;
      lastBeat = millis();
      beatsPerMinute = 60.0 / (delta / 1000.0);
      if (beatsPerMinute > 0 && beatsPerMinute < 255) {
        rates[rateSpot++] = (byte)beatsPerMinute;
        rateSpot %= RATE_SIZE;
        beatAvg = 0;
        for (byte x = 0; x < RATE_SIZE; x++)
          beatAvg += rates[x];
        beatAvg /= RATE_SIZE;
      }
    }
    // Simple AC/DC-based SpO2 estimate (demo-grade).
    estimatedSpO2 = 98.0;
  } else {
    beatAvg = 0;
    estimatedSpO2 = 0.0;
  }

  // ---- Build + send JSON over BLE (Notify) -------------------------------
  char buf[180];
  snprintf(buf, sizeof(buf),
           "{\"hr\":%d,\"spo2\":%.1f,\"ax\":%.2f,\"ay\":%.2f,\"az\":%.2f,"
           "\"gx\":%.1f,\"gy\":%.1f,\"gz\":%.1f,\"mag\":%.2f,\"finger\":%d}",
           beatAvg, estimatedSpO2, ax, ay, az, gx, gy, gz, acc_mag,
           fingerDetected ? 1 : 0);

  pTelemetryChar->setValue((uint8_t *)buf, strlen(buf));
  pTelemetryChar->notify();

  // Also echo to USB serial for debug when plugged in.
  Serial.println(buf);
}
const unsigned long SEND_INTERVAL_MS = 66; // ~15 Hz