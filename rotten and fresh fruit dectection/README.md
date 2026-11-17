# Fruit Freshness Detector 🍎🔍

A beginner-friendly, technically rigorous Computer Vision and Deep Learning system that classifies visible fruit condition as **FRESH**, **ROTTEN**, or **UNCERTAIN** using a trained **MobileNetV2** Convolutional Neural Network (CNN) and **ONNX** runtime.

Designed for **Windows laptops** (with integrated or USB webcams) and architected for standalone edge deployment on **Raspberry Pi 5** with **Camera Module 3**.

---

> [!IMPORTANT]
> **Visual Appearance Classifier Only**:
> This system evaluates visible external surface characteristics. It does **not** provide food-safety certification, does not determine internal chemical freshness or microbiological safety, and does not predict remaining shelf life or edibility.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Key Features](#key-features)
3. [Project Directory Structure](#project-directory-structure)
4. [Prerequisites & Requirements](#prerequisites--requirements)
5. [Installation & Setup (Windows)](#installation--setup-windows)
6. [Hardware Verification](#step-1-hardware-verification)
7. [Dataset Collection & Isolation Rules](#step-2-dataset-collection)
8. [Dataset Integrity Audit](#step-3-dataset-audit)
9. [Model Training & ONNX Export](#step-4-model-training)
10. [Model Evaluation & Rotten Recall](#step-5-evaluation)
11. [Live Camera Inspection (Scan Mode)](#step-6-live-camera-inspection)
12. [CSV Scan Logging](#csv-scan-logging)
13. [Raspberry Pi 5 Deployment](#raspberry-pi-5-deployment)
14. [Troubleshooting Guide](#troubleshooting-guide)
15. [Limitations & Roadmap](#limitations--roadmap)

---

## System Architecture

```
                                INFERENCE PIPELINE
Camera Feed (Webcam / Pi Cam 3)
      │
      ▼
Square Crop (Centered 80% ROI)
      │
      ▼
Image Preprocessing (224x224 RGB, Float32, ImageNet Normalization, CHW)
      │
      ▼
MobileNetV2 ONNX Model (OpenCV DNN Engine)
      │
      ▼
Softmax Probabilities [p_fresh, p_rotten]
      │
      ▼
Certainty Threshold Filter (Default: 80%)
     ├── Confidence >= 0.80 ──► FRESH or ROTTEN
     └── Confidence < 0.80  ──► UNCERTAIN
      │
      ▼
Presentation UI + Batch Counter Update + CSV Log Write (logs/scans.csv)
```

```
                                TRAINING PIPELINE
Dataset (data/train & data/val)
      │
      ▼
Dataset Audit (audit_dataset.py: Zero-Leakage & Specimen Isolation)
      │
      ▼
MobileNetV2 Transfer Learning (PyTorch + Torchvision Pretrained Weights)
      │
      ▼
Best Validation Checkpoint (artifacts/best_model.pt)
      │
      ▼
ONNX Export + Numerical Parity Verification (PyTorch vs OpenCV DNN atol=1e-3)
      │
      ▼
Artifacts Ready: artifacts/fruit.onnx + metadata.json
```

---

## Key Features

- **MobileNetV2 Transfer Learning**: Uses lightweight, highly efficient CNN architecture suited for laptop and embedded edge devices.
- **Strict 3-State Logic (`FRESH`, `ROTTEN`, `UNCERTAIN`)**: Ambiguous or low-confidence scans are flagged as `UNCERTAIN` instead of guessing.
- **Physical Specimen Isolation**: Strict dataset hygiene prevents the same physical fruit from appearing across train, val, or test splits (Zero Data Leakage).
- **Mandatory Numerical Parity Check**: Automatically verifies that PyTorch training predictions and OpenCV ONNX inference match within strict tolerance (`atol=1e-3`).
- **Batch Rotten Spoilage Alerting**: Configurable batch warning alerts the operator when rotten item counts exceed a threshold (e.g., 3 rotten fruits).
- **100% Offline & Local**: No internet connection, cloud services, or API keys required after initial installation.

---

## Project Directory Structure

```
fruit-freshness/
│
├── README.md                  # Master documentation & guide
├── QUICKSTART.md              # 10-step beginner quickstart path
├── PROJECT_SPEC.md            # School innovation & technical specification
├── SAFETY_AND_LIMITATIONS.md  # Critical safety disclaimers & operational limitations
├── DATASET_GUIDE.md           # Dataset collection, lighting, and isolation manual
├── PI_SETUP.md                # Raspberry Pi 5 & Camera Module 3 setup guide
├── requirements.txt           # Pinned Python package dependencies
├── .gitignore                 # Standard repository ignores
│
├── common.py                  # Core utilities, preprocessing, Camera & Predictor classes
├── app.py                     # Main interactive application (collect & scan modes)
├── train.py                   # PyTorch MobileNetV2 training & evaluation pipeline
├── evaluate.py                # Standalone evaluation & confusion matrix reporter
├── audit_dataset.py           # Dataset integrity, duplicate & specimen isolation auditor
├── camera_test.py             # Hardware diagnostic tool for webcam and Pi camera
├── export_model.py            # Checkpoint to ONNX exporter & numerical parity verifier
│
├── artifacts/                 # Saved model weights, ONNX files, metadata, and graphs
│   └── README.md
│
├── data/                      # Dataset splits (train/val/test x fresh/rotten)
│   ├── README.md
│   ├── train/ {fresh, rotten}
│   ├── val/   {fresh, rotten}
│   └── test/  {fresh, rotten}
│
├── logs/                      # Scan event logs
│   ├── README.md
│   └── scans.csv
│
├── reports/                   # Quantitative evaluation JSON reports
│   └── README.md
│
├── tests/                     # Automated unit test suite
│   ├── test_common.py
│   ├── test_dataset.py
│   ├── test_metadata.py
│   └── test_preprocessing.py
│
└── pi/                        # Raspberry Pi specific scripts & GPIO configs
    ├── README.md
    ├── install_pi.sh
    └── gpio_config.py
```

---

## Prerequisites & Requirements

- **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu / Raspberry Pi OS 64-bit).
- **Python**: Python 3.12 or 3.13.
- **Hardware**: Any standard laptop webcam or external USB webcam (Raspberry Pi 5 + Camera Module 3 for Pi deployment).

---

## Installation & Setup (Windows)

Open **PowerShell** or **Command Prompt** and navigate to your project workspace:

### 1. Create and Activate Virtual Environment
```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
```
*(If using Python 3.13, you can run `python -m venv .venv`)*

### 2. Upgrade Pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Run Automated Unit Tests
```powershell
python -m pytest tests/ -v
```
*(All 16 tests should pass).*

---

## Complete Step-by-Step Walkthrough

### Step 1: Hardware Verification
Test that your webcam can be opened and inspect the crop framing:
```powershell
python camera_test.py
```
- A preview window will display your webcam feed with a centered green inspection square.
- Verify that your camera feed is clear and responsive.
- Press **Q** to exit.

---

### Step 2: Dataset Collection
The built-in collector tool crops only the inspection square and enforces unique physical fruit naming:

```powershell
# 1. Collect FRESH specimen for training
python app.py collect --split train --label fresh --id apple001

# 2. Collect ROTTEN specimen for training
python app.py collect --split train --label rotten --id apple101

# 3. Collect independent FRESH specimens for validation and test
python app.py collect --split val --label fresh --id apple002
python app.py collect --split test --label fresh --id apple003

# 4. Collect independent ROTTEN specimens for validation and test
python app.py collect --split val --label rotten --id apple102
python app.py collect --split test --label rotten --id apple103
```

**Controls in Collection Window:**
- **S**: Save current square crop (`apple001__<unique-id>.jpg`).
- Rotate the fruit slightly between presses to capture multiple angles.
- **Q**: Finish capturing this specimen and exit.

---

### Step 3: Dataset Audit
Before training, audit your dataset to ensure no corrupt images exist and that physical specimens are strictly isolated:
```powershell
python audit_dataset.py
```
The audit verifies:
- All 6 split folders exist (`train/val/test` × `fresh/rotten`).
- Images are valid and readable.
- **Zero Leakage**: No fruit ID (e.g. `apple001`) appears in more than one split.
- **Duplicate Detection**: No identical image files cross split boundaries.

---

### Step 4: Model Training
Train the MobileNetV2 classifier on your collected dataset:
```powershell
python train.py --epochs 15 --batch-size 16
```

What this does automatically:
1. Loads pretrained ImageNet weights from Torchvision.
2. Warms up the 2-class classification head, then fine-tunes upper layers.
3. Saves the best validation checkpoint to `artifacts/best_model.pt`.
4. Saves loss/accuracy curves to `artifacts/training_graph.png`.
5. Exports the model to `artifacts/fruit.onnx`.
6. Executes **PyTorch vs ONNX numerical verification** (`atol=1e-3`).
7. Writes `artifacts/metadata.json` and evaluates the validation split.

---

### Step 5: Evaluation
Evaluate model accuracy, confusion matrix, and critical **Rotten Recall**:

```powershell
# Evaluate Validation split
python train.py --evaluate val --threshold 0.80

# Evaluate Test split (unseen evaluation data)
python train.py --evaluate test --threshold 0.80
```

#### Why Rotten Recall is the Critical Metric:
In fruit quality inspection, missing a rotten fruit (**False Negative**) allows rot to spread to other fruit in storage. While overall accuracy is informative, **Rotten Recall** specifically measures the percentage of visibly rotten items caught by the model:

$$\text{Rotten Recall} = \frac{\text{True Rotten Detections}}{\text{Total Actual Rotten Specimens}}$$

---

### Step 6: Live Camera Inspection
Launch the presentation inspection console:
```powershell
python app.py scan --threshold 0.80 --alert-count 3
```

**Interactive Controls:**
- **S** (*Scan*): Takes one deliberate frame capture, crops the square, runs CNN inference, displays score and verdict, updates batch counts, and logs to `logs/scans.csv`.
- **B** (*New Batch*): Resets batch counters and generates a new Batch ID without erasing historical CSV records.
- **Q** (*Quit*): Closes camera and exits cleanly.

---

## CSV Scan Logging

Every deliberate scan recorded with **S** appends a row to `logs/scans.csv`:

```csv
time,batch,result,fresh_score,rotten_score,threshold,inference_ms
2026-09-26T10:30:00.123456+05:30,BATCH-3A9F1B,FRESH,0.9412,0.0588,0.80,38.2
2026-09-26T10:30:15.654321+05:30,BATCH-3A9F1B,ROTTEN,0.0310,0.9690,0.80,36.9
2026-09-26T10:30:32.789123+05:30,BATCH-3A9F1B,UNCERTAIN,0.6120,0.3880,0.80,37.1
```

---

## Raspberry Pi 5 Deployment

The Raspberry Pi 5 acts as a standalone inference appliance using the model trained on your laptop.

```bash
# 1. On Raspberry Pi, run automated setup
chmod +x pi/install_pi.sh
./pi/install_pi.sh

# 2. Copy artifacts/fruit.onnx and artifacts/metadata.json to the Pi
# 3. Test Pi Camera Module 3
python camera_test.py --source pi

# 4. Launch live scanner with optional GPIO status LEDs
python app.py scan --source pi --gpio --threshold 0.80 --alert-count 3
```

See [PI_SETUP.md](PI_SETUP.md) for full wiring schematics and pinouts.

---

## Troubleshooting Guide

| Problem | Likely Cause | Recommended Fix |
|:---|:---|:---|
| **Cannot open webcam** | Webcam is locked by Zoom, Teams, Camera app, or browser | Close all camera-using applications and check Windows Camera privacy settings. |
| **No trained model found** | Model has not been trained yet | Collect images in `data/` and run `python train.py --epochs 15`. |
| **Audit fails: Physical fruit leakage** | Same fruit ID appears in multiple splits | Ensure all views of a physical fruit (e.g. `apple001`) are only in `train`, `val`, OR `test`. |
| **Audit fails: Duplicate image** | Same image copied into different folders | Remove duplicate image files across splits. |
| **`--gpio` rejected on laptop** | GPIO hardware only exists on Raspberry Pi | Run without `--gpio` on PC, or specify `--source pi` on Raspberry Pi. |

---

## Limitations & Roadmap

### Current Limitations:
1. **Single Variety Focus**: Default model is calibrated for one fruit variety at a time (e.g. Apple).
2. **Not an Object Detector**: Non-fruit objects or empty backgrounds inside the square will receive a classification score. Always place one fruit inside the square.
3. **Visual-Only**: Cannot identify internal rot, fungal mycotoxins, chemical residue, or bacterial contamination not visible on the peel.

### Future Roadmap:
- Multi-fruit variety classifier / auto-fruit selector
- Multi-angle 3D view fusion
- Background / empty tray rejection model
- Physical sorting mechanism integration (actuators/conveyors)
- Model confidence calibration using Temperature Scaling
