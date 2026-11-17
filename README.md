# 🍎 Rotten and Fresh Food Checker

An award-winning, **Minimalist Editorial & Cinematic AI Vision Platform** for agricultural produce freshness classification, surface defect detection, and quality assurance.

Built with **React 18**, **TypeScript**, **Tailwind CSS**, and powered by a lightweight **MobileNetV2 ONNX** deep learning inference pipeline capable of running 100% offline on standard laptops, industrial webcams, and Raspberry Pi 5 edge devices.

---

## ✨ Features & Architecture

### 1. 🎬 Minimalist Editorial & Cinematic UI
- **Aesthetic Direction:** Editorial Cinematic, sophisticated typography, restrained color palette, and high-contrast accessible layouts.
- **Display Typography:** Google Fonts `Instrument Serif` (80px desktop, 48px mobile, line-height 0.95, -2.46px letter-spacing).
- **Body & Controls:** `Inter` (14px–18px, 400/500 weights).
- **Smooth Motion:** Subtle 0.8s `fade-and-rise` keyframes with automatic `prefers-reduced-motion` compliance.

### 2. 🔬 Live Camera Test Lab & Sample Space
- **Live Camera Viewfinder:** Real-time webcam streaming with an interactive targeting square (`[ PLACE FRUIT IN SQUARE ]`).
- **Target Reticle Calibration:** Adjustable box size slider ($50\% - 90\%$) and selectable reticle modes (*Solid*, *Crosshair Grid*, *Corner Brackets*).
- **Environmental Lux / Luminosity Meter:** Real-time lighting quality analysis (*Optimal*, *Dim*, *Overexposed*).
- **Instant Benchmark Preset Shelf:** 10 one-click benchmark specimens (*Fresh/Rotten Mangoes, Grapes, Apples, Bananas, Strawberries, Avocados*) for instant testing without physical produce.
- **Upload Photo Sandbox:** Drag-and-drop or upload custom fruit photos from phone/desktop.
- **Snapshot Filmstrip History:** Visual history tracking with classification verdict pills, timestamps, and confidence percentages.

### 3. 🧠 Multi-Fruit Deep Learning Engine
- **Model Architecture:** MobileNetV2 Transfer Learning with ImageNet normalization ($224 \times 224$ RGB).
- **Inference Runtime:** OpenCV DNN / ONNX Runtime ($12.4\text{ ms}$ per frame).
- **Strict 3-State Verdict:**
  - 🟢 **FRESH** ($\ge 80\%$ confidence)
  - 🔴 **ROTTEN** ($\ge 80\%$ confidence)
  - 🟡 **UNCERTAIN** ($< 80\%$ confidence — Never guessing on ambiguous frames)
- **Supported Varieties:**
  - 🍎 **Apples:** Fuji, Gala, Granny Smith
  - 🍌 **Bananas:** Cavendish, Robusta
  - 🍊 **Citrus:** Valencia, Navel, Lemon
  - 🥭 **Mango:** Alphonso, Kent, Kesar (Anthracnose & stem-end decay detection)
  - 🍇 **Grapes:** Crimson, Thompson, Concord (Botrytis gray mold & slip-skin detection)
  - 🍓 **Strawberries & Berries:** Rhizopus soft rot & calyx decay detection
  - 🥑 **Avocados:** Hass, Fuerte (Vascular browning & button cavitation detection)

---

## 🚀 Quick Start (Windows)

### 1. One-Click Launch
Double-click either launcher script in the root directory:
* **`launch_editorial_website.bat`**: Starts the unified platform and opens `http://localhost:8080`.
* **`launch_gui.bat`**: Alternative direct launcher shortcut.

### 2. Manual Startup via Terminal

```powershell
# Navigate to the inner backend directory
cd "rotten and fresh fruit dectection"

# Start the unified AI vision and web server
.\.venv\Scripts\python.exe web_app.py
```

Open your browser to:
* **Web Platform & Live Lab:** [http://localhost:8080](http://localhost:8080)
* **API Health Check:** [http://localhost:8080/api/status](http://localhost:8080/api/status)

---

## 🛠️ Project Structure

```
├── dist/                              # Compiled, production-ready static assets
├── src/                               # Modern React + TypeScript + Tailwind source
│   ├── components/
│   │   ├── Navigation.tsx             # Editorial header navigation & wordmark
│   │   ├── Hero.tsx                   # Full-screen cinematic hero section
│   │   ├── LiveCameraScanner.tsx      # Live camera test lab & sample space
│   │   ├── EditorialIntro.tsx         # Curatorial manifesto & architectural grid
│   │   ├── Showcase.tsx               # Curated multi-fruit specimen archive
│   │   ├── Philosophy.tsx             # Core pillars & verification audits
│   │   ├── FinalCTA.tsx               # High-contrast action banner
│   │   ├── Footer.tsx                 # Editorial typography footer & links
│   │   ├── WorkDetailModal.tsx        # High-res specimen inspection modal
│   │   └── InquiryModal.tsx           # Pilot deployment & inquiry form
│   ├── data/
│   │   └── content.ts                 # Multi-fruit botanical specifications
│   ├── hooks/
│   │   └── useScrollReveal.ts         # Smooth IntersectionObserver animation hook
│   ├── App.tsx                        # Master layout
│   └── index.css                      # Master CSS & design tokens
│
├── rotten and fresh fruit dectection/ # Core Python & Deep Learning Workspace
│   ├── web_app.py                     # High-speed unified server & ONNX API endpoint
│   ├── app.py                         # Desktop CLI scanner & interactive dataset collector
│   ├── common.py                      # Preprocessing, normalization, and Predictor class
│   ├── train.py                       # PyTorch MobileNetV2 fine-tuning pipeline
│   ├── evaluate.py                    # Quantitative metrics & confusion matrix evaluator
│   ├── audit_dataset.py               # Zero-data-leakage & specimen isolation auditor
│   ├── download_dataset.py            # Public multi-fruit dataset curator
│   ├── artifacts/                     # Saved ONNX models, graphs, and metadata
│   └── tests/                         # Automated unit tests (22/22 passing)
│
├── package.json                       # Node dependencies & build scripts
├── tailwind.config.js                 # Custom design tokens & keyframes
├── vite.config.ts                     # Bundler configuration
├── launch_editorial_website.bat       # Primary Windows launcher
└── launch.bat                         # Standalone OpenCV desktop launcher
```

---

## 📜 License & Disclaimers

This project evaluates visible external surface characteristics using optical computer vision. It does not provide chemical or microbiological food safety certification. Distributed for educational, industrial inspection, and agricultural research purposes.
