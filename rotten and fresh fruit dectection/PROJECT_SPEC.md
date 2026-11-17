# Project Specification — Fruit Freshness Detector 🍎📊

## 1. Executive Summary & Problem Statement

### The Problem:
Manual inspection of fruit for spoilage in cafeterias, retail stores, and small-scale packing lines is subjective, labor-intensive, and prone to operator fatigue. Early visual signs of decay (such as bruising, localized mold patches, soft rot, and skin discoloration) are frequently missed, leading to cross-contamination of adjacent stored produce.

### Proposed Solution:
An edge-ready, computer-vision system that captures live camera frames, extracts a standardized Region of Interest (ROI), and classifies visible fruit surface appearance using a trained **MobileNetV2** Convolutional Neural Network (CNN).

The system outputs a 3-state verdict (**FRESH**, **ROTTEN**, or **UNCERTAIN**), logs inspection records to a CSV file, and alerts operators when rotten produce counts exceed a batch threshold.

---

## 2. Technical System Architecture

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  CAMERA SOURCE  │──────►│   SQUARE CROP   │──────►│  PREPROCESSING  │
│ Webcam / PiCam3 │       │ 80% Center ROI  │       │ 224x224 RGB CHW │
└─────────────────┘       └─────────────────┘       └────────┬────────┘
                                                             │
                                                             ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   BATCH ALERT   │◄──────│ THRESHOLD LOGIC │◄──────│  MOBILENETV2    │
│ & CSV LOGGING   │       │ Score >= 0.80   │       │   ONNX ENGINE   │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

### Key Components:
1. **Camera Layer**: Standard OpenCV video capture (`webcam`) or native Raspberry Pi Picamera2 (`pi`).
2. **Crop & Geometry Layer**: Centered square crop occupying ~80% of the minor frame dimension.
3. **Preprocessing Pipeline**: BGR to RGB color conversion, bilinear resize to $224 \times 224$, float32 scaling $[0, 1]$, and standard ImageNet normalization ($\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225]$).
4. **Machine Learning Classifier**: Pretrained MobileNetV2 with custom 2-class classification head (`fresh` vs `rotten`).
5. **Inference Engine**: OpenCV DNN executing the exported `fruit.onnx` model on CPU.
6. **Decision & Threshold Logic**: Softmax output with a configurable confidence threshold (default 80%). Predictions below threshold are marked `UNCERTAIN`.
7. **Batch & Audit Logging**: Unique batch identification, running category counters, batch rotten alert trigger, and persistent CSV event logging.

---

## 3. Machine Learning Methodology

### Neural Network Backbone:
- **Architecture**: MobileNetV2 (Sandler et al., 2018).
- **Rationale**: Inverted residual blocks with linear bottlenecks provide high accuracy with minimal floating-point operations (FLOPs) and low memory footprint (~14 MB), making it ideal for real-time CPU execution on laptops and Raspberry Pi 5.
- **Input Dimensions**: $(1, 3, 224, 224)$ float32 tensor.
- **Output Layer**: Linear layer with 2 logits followed by softmax.

### Transfer Learning Strategy:
- **Phase 1 (Warmup)**: Feature extractor frozen; only the custom classification head is trained for 3 epochs using AdamW optimizer ($\text{lr} = 10^{-3}$).
- **Phase 2 (Fine-Tuning)**: Upper feature blocks (Block 14+) unfrozen and fine-tuned with reduced learning rate ($\text{lr} = 2 \times 10^{-4}$) and Cosine Annealing learning rate scheduling.

### Model Export & Numerical Parity:
After training, the model is exported to ONNX format. A mandatory parity test verifies that OpenCV DNN inference produces output probabilities matching PyTorch within strict tolerance ($\text{atol} = 10^{-3}, \text{rtol} = 10^{-3}$).

---

## 4. Evaluation Metrics & The Critical Role of Rotten Recall

Evaluation computes standard classification metrics, with particular emphasis on **Rotten Recall**:

$$\text{Forced Accuracy} = \frac{TP + TN}{Total}$$

$$\text{Rotten Recall} = \frac{TP_{\text{rotten}}}{TP_{\text{rotten}} + FN_{\text{rotten}}}$$

$$\text{Rotten Precision} = \frac{TP_{\text{rotten}}}{TP_{\text{rotten}} + FP_{\text{rotten}}}$$

$$\text{Accepted Accuracy} = \frac{\text{Correct Accepted Predictions}}{\text{Total Confident Predictions}}$$

### Why Rotten Recall is the Primary Objective:
In food logistics, a **False Positive** (a fresh apple flagged as rotten) costs the price of one apple. However, a **False Negative** (a rotten apple classified as fresh) can cause mold and fungal spores to rot an entire box or storage crate. Therefore, high Rotten Recall is the central metric to evaluate during validation.

---

## 5. Hardware Specifications

| Component | Laptop Milestone (Phase 1) | Raspberry Pi 5 Milestone (Phase 2) |
|:---|:---|:---|
| **Host System** | Windows 10/11 Laptop or PC | Raspberry Pi 5 (4GB or 8GB RAM) |
| **Camera** | Built-in webcam / USB webcam (640×480) | Raspberry Pi Camera Module 3 (Standard) |
| **Autofocus** | Fixed / Autofocus via USB UVC | Continuous Autofocus via `libcamera` |
| **Inference Engine** | OpenCV DNN CPU | OpenCV DNN CPU / ONNX Runtime |
| **Indicators** | Graphical OpenCV On-Screen Console | OpenCV UI + Optional GPIO LEDs (G/Y/R) |
| **Logging** | Local `logs/scans.csv` | Local `logs/scans.csv` |

---

## 6. Scientific Limitations & Disclaimers

1. **Visual Appearance Only**: The CNN analyzes pixel reflectance and visible surface patterns. It cannot detect internal watercore, internal browning, chemical pesticides, bacterial pathogens (e.g. *Salmonella*, *E. coli*), or mycotoxins.
2. **Single-Variety Model**: The initial model is trained and validated on a single fruit variety (e.g., Apple). Generalization to citrus, bananas, or stone fruit requires separate validated models.
3. **No Unknown Object Rejection**: The baseline system assumes a supported fruit is present in the crop area. It is not an open-world object detector.
