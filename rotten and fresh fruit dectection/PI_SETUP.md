# Raspberry Pi 5 & Camera Module 3 Setup Guide 🍓📷

This guide walks through deploying the **Fruit Freshness Detector** as a standalone, offline edge device using a **Raspberry Pi 5** and **Raspberry Pi Camera Module 3**.

---

## 1. Hardware Bill of Materials

| Item | Component | Specification |
|:---|:---|:---|
| **SBC** | Raspberry Pi 5 | 4 GB or 8 GB RAM |
| **Camera** | Raspberry Pi Camera Module 3 | Standard visible-light version (Auto-Focus) |
| **CSI Cable** | Pi 5 Camera Ribbon Cable | 22-pin to 15-pin FPC cable |
| **Power Supply** | Official 27W USB-C PSU | 5.1V / 5.0A PD |
| **Storage** | MicroSD Card | 32 GB+ Class 10 / A2 (Raspberry Pi OS 64-bit Bookworm) |
| **Optional LEDs** | 3x 5mm LEDs | 1x Green, 1x Yellow, 1x Red |
| **Resistors** | 3x Current-limiting resistors | 220Ω to 330Ω (1/4W) |
| **Prototyping** | Breadboard & Jumper wires | Male-to-Female & Male-to-Male jumpers |

> [!IMPORTANT]
> **Use the Standard Visible-Light Camera**:
> Do **NOT** use the NoIR (Infrared) Camera Module 3. The freshness classification model relies on natural visible color spectrum representation.

---

## 2. GPIO Status LED Wiring Schematic

Connect the 3 indicator LEDs to the Raspberry Pi 40-pin header:

```
Raspberry Pi 5 Header Pinout:
  Pin 11 (GPIO 17) ────► [ 220Ω Resistor ] ────► [ Green LED Anode (+) ] ────┐
  Pin 13 (GPIO 27) ────► [ 220Ω Resistor ] ────► [ Yellow LED Anode (+) ] ───┤
  Pin 15 (GPIO 22) ────► [ 220Ω Resistor ] ────► [ Red LED Anode (+) ] ──────┤
                                                                             │
  Pin 9, 14, or 20 (GND) ◄────────────────────── [ All LED Cathodes (-) ] ───┘
```

| Signal / Verdict | LED Color | BCM GPIO | Physical Header Pin | Resistor Value | Ground Pin |
|:---|:---|:---:|:---:|:---:|:---:|
| **FRESH** | Green | GPIO 17 | Pin 11 | 220Ω | Pin 9 (GND) |
| **UNCERTAIN** | Yellow / Amber | GPIO 27 | Pin 13 | 220Ω | Pin 14 (GND) |
| **ROTTEN** | Red | GPIO 22 | Pin 15 | 220Ω | Pin 20 (GND) |

> [!CAUTION]
> - Always connect resistors in series with LEDs to protect the Raspberry Pi GPIO pins from over-current damage.
> - Never connect buzzers, relay coils, or motors directly to GPIO pins without a dedicated driver circuit.

---

## 3. Software Installation on Raspberry Pi OS (Bookworm 64-bit)

### Step 1: Clone Repository and Run Automated Setup
Open a terminal on your Raspberry Pi:
```bash
git clone <your-repository-url> fruit-freshness
cd fruit-freshness
chmod +x pi/install_pi.sh
./pi/install_pi.sh
```

### Step 2: Transfer Trained Model Artifacts from Your Laptop
Because model training is computationally intensive, train the model on your laptop and transfer the verified ONNX model and metadata to the Pi:

On your **Laptop**, run:
```powershell
# Windows PowerShell (replace 'pi-ip-address' with your Pi's local IP address)
scp artifacts/fruit.onnx pi@<pi-ip-address>:~/fruit-freshness/artifacts/
scp artifacts/metadata.json pi@<pi-ip-address>:~/fruit-freshness/artifacts/
```

---

## 4. Hardware Verification & Live Scanning on Pi

### Step 3: Test Camera Module 3
```bash
source .venv/bin/activate
python camera_test.py --source pi
```
*Verify that the camera initializes with continuous autofocus and displays a smooth 30 FPS video feed. Press **Q** to exit.*

### Step 4: Run Real-Time Scanner with GPIO LEDs
```bash
source .venv/bin/activate
python app.py scan --source pi --gpio --threshold 0.80 --alert-count 3
```

**Controls in Scanner:**
- **S**: Capture frame, crop square ROI, run MobileNetV2 ONNX inference, light up status LED (Green / Yellow / Red), display score on screen, update batch counts, and append record to `logs/scans.csv`.
- **B**: Reset batch counter and turn off LEDs.
- **Q**: Clean exit (safely releases camera and extinguishes all LEDs).
