#!/usr/bin/env bash
# =============================================================================
# Fruit Freshness Detector - Raspberry Pi 5 Setup Script
# =============================================================================
# Configures Raspberry Pi OS (64-bit Bookworm), Picamera2, OpenCV, and ONNX
# runtime environment for local offline inference.
# =============================================================================

set -e

echo "====================================================================="
echo "FRUIT FRESHNESS DETECTOR - RASPBERRY PI 5 SETUP"
echo "====================================================================="

# 1. Update system packages
echo "[*] Updating apt package repository..."
sudo apt-get update -y

# 2. Install native system dependencies and camera tools
echo "[*] Installing Python, Picamera2, libcamera, and OpenCV system packages..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-picamera2 \
    python3-libcamera \
    libcamera-apps \
    python3-opencv \
    python3-gpiozero \
    python3-rpi.gpio \
    python3-numpy \
    python3-pil

# 3. Create virtual environment with access to system packages
echo "[*] Creating virtual environment (.venv) with --system-site-packages..."
if [ ! -d ".venv" ]; then
    python3 -m venv --system-site-packages .venv
fi

# 4. Activate virtual environment and install ML inference dependencies
echo "[*] Activating virtual environment and installing python dependencies..."
source .venv/bin/activate
pip install --upgrade pip
pip install onnx==1.18.0 pytest>=8.0.0

echo ""
echo "====================================================================="
echo "[+] Raspberry Pi environment setup completed successfully!"
echo "====================================================================="
echo ""
echo "Next Steps:"
echo "  1. Copy your trained 'artifacts/fruit.onnx' and 'artifacts/metadata.json'"
echo "     from your training PC to this Raspberry Pi."
echo "  2. Test Camera Module 3:"
echo "     source .venv/bin/activate"
echo "     python camera_test.py --source pi"
echo "  3. Launch live freshness scanner with optional GPIO LEDs:"
echo "     python app.py scan --source pi --gpio --threshold 0.80 --alert-count 3"
echo "====================================================================="
