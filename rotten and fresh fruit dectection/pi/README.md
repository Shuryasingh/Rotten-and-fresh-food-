# Raspberry Pi 5 Deployment Guide

This directory contains configuration scripts and hardware interface modules for deploying the **Fruit Freshness Detector** on a standalone **Raspberry Pi 5** with **Camera Module 3**.

---

## System Role & Architecture

```
[ Training PC / Laptop ]                     [ Standalone Raspberry Pi 5 ]
┌─────────────────────────┐                 ┌─────────────────────────────┐
│ 1. Collect Dataset      │                 │ 1. Camera Module 3 (AF)     │
│ 2. Audit Dataset        │                 │ 2. Picamera2 Frame Grabber  │
│ 3. Train MobileNetV2    │   fruit.onnx    │ 3. OpenCV DNN Inference     │
│ 4. Export & Verify ONNX ├────────────────►│ 4. Live Video UI            │
│ 5. Evaluate Split       │  metadata.json  │ 5. Optional Status LEDs     │
└─────────────────────────┘                 │ 6. Offline Local Logging    │
                                            └─────────────────────────────┘
```

The Raspberry Pi acts as an **offline inference appliance**:
- Training is performed on your PC/laptop.
- The verified `artifacts/fruit.onnx` and `artifacts/metadata.json` are copied to the Pi.
- The Pi runs real-time inference locally without requiring cloud or internet connectivity.

---

## Hardware Requirements

1. **Raspberry Pi 5** (4 GB or 8 GB RAM).
2. **Raspberry Pi Camera Module 3** (Standard visible-light version with continuous autofocus).
   > [!IMPORTANT]
   > Do NOT use NoIR (Infrared) camera versions, as the freshness classifier relies heavily on visible color spectrum representation.
3. **Optional Status LEDs**:
   - 1x Green LED (Fresh)
   - 1x Yellow / Amber LED (Uncertain)
   - 1x Red LED (Rotten)
   - 3x 220Ω – 330Ω Current-limiting resistors
   - Breadboard and jumper wires

---

## GPIO Status LED Wiring Diagram

Connect the LEDs to the Raspberry Pi 40-pin GPIO header:

| Indicator | Signal | GPIO Pin (BCM) | Physical Header Pin | Resistor | Ground Pin |
|:---|:---|:---:|:---:|:---:|:---:|
| **Fresh** | Green LED | GPIO 17 | Pin 11 | 220Ω | Pin 9 (GND) |
| **Uncertain** | Yellow LED | GPIO 27 | Pin 13 | 220Ω | Pin 14 (GND) |
| **Rotten** | Red LED | GPIO 22 | Pin 15 | 220Ω | Pin 20 (GND) |

```
Raspberry Pi Header
  Pin 11 (GPIO 17) ───[ 220Ω ]─── (Anode +) [Green LED] (Cathode -) ─── Pin 9 (GND)
  Pin 13 (GPIO 27) ───[ 220Ω ]─── (Anode +) [Yellow LED] (Cathode -) ── Pin 14 (GND)
  Pin 15 (GPIO 22) ───[ 220Ω ]─── (Anode +) [Red LED] (Cathode -) ───── Pin 20 (GND)
```

> [!CAUTION]
> - Always use current-limiting resistors (220Ω – 330Ω) in series with each LED.
> - Do not connect buzzers, relay coils, or motors directly to the GPIO pins.

---

## Installation & Setup on Pi

1. **Clone repository and run automated setup**:
   ```bash
   chmod +x pi/install_pi.sh
   ./pi/install_pi.sh
   ```

2. **Copy trained model artifacts from laptop**:
   Transfer `fruit.onnx` and `metadata.json` into `artifacts/`:
   ```bash
   scp user@laptop-ip:fruit-freshness/artifacts/fruit.onnx artifacts/
   scp user@laptop-ip:fruit-freshness/artifacts/metadata.json artifacts/
   ```

3. **Verify Camera Module 3 connectivity**:
   ```bash
   source .venv/bin/activate
   python camera_test.py --source pi
   ```

4. **Launch Live Inspection Scanner**:
   ```bash
   # Basic scanner (screen only)
   python app.py scan --source pi --threshold 0.80

   # Scanner with physical GPIO status LEDs
   python app.py scan --source pi --gpio --threshold 0.80 --alert-count 3
   ```
