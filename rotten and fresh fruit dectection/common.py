"""Common utilities, camera abstraction, image preprocessing, and inference predictor.

This module is shared across collection, live scanning, training, export, and evaluation.
All filesystem paths are resolved relative to the project root directory.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FruitFreshness")

# -----------------------------------------------------------------------------
# Project Path Configuration
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_ONNX_PATH = ARTIFACTS_DIR / "fruit.onnx"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
MODEL_PT_PATH = ARTIFACTS_DIR / "best_model.pt"
SCAN_LOG_PATH = LOGS_DIR / "scans.csv"

# -----------------------------------------------------------------------------
# Model & Application Constants
# -----------------------------------------------------------------------------
CLASSES = ["fresh", "rotten"]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}

SIZE = 224
DEFAULT_THRESHOLD = 0.80
DEFAULT_ALERT_COUNT = 3
DEFAULT_SUPPORTED_FRUIT = "APPLE"
SUPPORTED_FRUITS = ["APPLE", "BANANA", "ORANGE", "MANGO", "GRAPES", "STRAWBERRY", "AVOCADO"]

# Standard ImageNet normalization coefficients (matching MobileNetV2)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Fruit ID validation regex: letters, numbers, and hyphens only
FRUIT_ID_REGEX = re.compile(r"^[A-Za-z0-9-]+$")


# -----------------------------------------------------------------------------
# Image Geometry & Preprocessing
# -----------------------------------------------------------------------------
def square(frame: np.ndarray, coverage: float = 0.80) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Extract a centered square inspection area from a camera frame.

    Args:
        frame: Input image array from camera (H x W x C).
        coverage: Fraction of the smaller dimension to occupy (default 80%).

    Returns:
        cropped_image: The cropped square region of the image.
        (x, y, side): Coordinates and side length of the square bounding box.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Invalid frame supplied to square crop function.")

    h, w = frame.shape[:2]
    side = int(min(h, w) * coverage)
    side = max(16, side)  # Ensure a minimum sensible crop size

    x = (w - side) // 2
    y = (h - side) // 2

    cropped = frame[y : y + side, x : x + side].copy()
    return cropped, (x, y, side)


def tensor_image(bgr: np.ndarray) -> np.ndarray:
    """Preprocess a BGR image crop for MobileNetV2 ONNX inference.

    Steps:
    1. Convert BGR -> RGB
    2. Resize to 224x224 using bilinear interpolation
    3. Convert to float32 and scale to [0.0, 1.0]
    4. Normalize with ImageNet mean and std
    5. Transpose HWC -> CHW format
    6. Return a contiguous 4D batch array of shape (1, 3, 224, 224)

    Args:
        bgr: Input BGR image crop as a numpy array.

    Returns:
        Contiguous float32 numpy array ready for CNN input.
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("Invalid image supplied to tensor_image preprocessing.")

    # 1. BGR -> RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 2. Resize to SIZE x SIZE (224x224)
    resized = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)

    # 3. Float32 scaling to [0.0, 1.0]
    img_float = resized.astype(np.float32) / 255.0

    # 4. Normalize with ImageNet statistics
    normalized = (img_float - IMAGENET_MEAN) / IMAGENET_STD

    # 5. Transpose HWC (224, 224, 3) to CHW (3, 224, 224)
    chw = np.transpose(normalized, (2, 0, 1))

    # 6. Add batch dimension (1, 3, 224, 224) as contiguous array
    batched = np.ascontiguousarray(chw[np.newaxis, ...], dtype=np.float32)
    return batched


# -----------------------------------------------------------------------------
# Metadata Helpers
# -----------------------------------------------------------------------------
def load_metadata(path: Path = METADATA_PATH) -> Dict[str, Any]:
    """Load and strictly validate model metadata from JSON.

    Args:
        path: Path to metadata.json file.

    Returns:
        Validated metadata dictionary.

    Raises:
        FileNotFoundError: If metadata file does not exist.
        ValueError: If metadata structure, classes, or image size are invalid.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model metadata not found at '{path}'.\n"
            "Problem: No trained model metadata exists in the workspace.\n"
            "Cause: Training has not yet been performed.\n"
            "Fix: Collect dataset images and run: python train.py --epochs 15 --batch-size 16"
        )

    with open(path, "r", encoding="utf-8") as f:
        try:
            meta = json.load(f)
        except json.JSONDecodeError as err:
            raise ValueError(f"Corrupt metadata JSON at '{path}': {err}") from err

    # Validate mandatory fields
    required_keys = ["classes", "size", "model", "format"]
    for key in required_keys:
        if key not in meta:
            raise ValueError(f"Metadata missing required key '{key}' in '{path}'.")

    if meta["classes"] != CLASSES:
        raise ValueError(
            f"Metadata class mismatch in '{path}'. Expected {CLASSES}, but got {meta['classes']}."
        )

    if int(meta["size"]) != SIZE:
        raise ValueError(
            f"Metadata image size mismatch in '{path}'. Expected {SIZE}, but got {meta['size']}."
        )

    return meta


def save_metadata(metadata: Dict[str, Any], path: Path = METADATA_PATH) -> None:
    """Save metadata dictionary to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# -----------------------------------------------------------------------------
# Camera Abstraction
# -----------------------------------------------------------------------------
class Camera:
    """Unified camera abstraction supporting standard USB/Laptop Webcams and Raspberry Pi Camera Module 3."""

    def __init__(
        self,
        source: str = "webcam",
        index: int = 0,
        width: int = 640,
        height: int = 480,
    ):
        self.source = source.lower()
        self.index = index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._picam2: Optional[Any] = None

        if self.source == "webcam":
            self._init_webcam()
        elif self.source == "pi":
            self._init_picamera()
        else:
            raise ValueError(f"Unsupported camera source: '{source}'. Choose 'webcam' or 'pi'.")

    def _init_webcam(self) -> None:
        """Initialize standard OpenCV video capture device."""
        # On Windows, cv2.CAP_DSHOW provides fast initialization and stable device index binding
        if sys.platform.startswith("win"):
            self._cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.index)

        if not self._cap.isOpened():
            # Fallback without DSHOW if first attempt failed
            self._cap = cv2.VideoCapture(self.index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam at index {self.index}.\n"
                "Problem: Camera device could not be opened.\n"
                "Cause: The camera may be in use by another application (Zoom, Teams, Camera App) "
                "or disabled in Windows Settings.\n"
                "Fix: Close all camera-using applications and verify webcam permissions."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _init_picamera(self) -> None:
        """Initialize Raspberry Pi Camera Module 3 via Picamera2."""
        try:
            from picam2 import Picamera2  # type: ignore
        except ImportError as err:
            raise ImportError(
                "Picamera2 is not installed.\n"
                "Problem: Cannot initialize Raspberry Pi camera.\n"
                "Cause: Picamera2 is only available on Raspberry Pi OS.\n"
                "Fix: When running on a laptop/PC, use '--source webcam'."
            ) from err

        try:
            self._picam2 = Picamera2()
            config = self._picam2.create_preview_configuration(
                main={"format": "XBGR8888", "size": (self.width, self.height)}
            )
            self._picam2.configure(config)
            self._picam2.start()
            # Enable continuous autofocus for Camera Module 3
            try:
                from libcamera import controls  # type: ignore
                self._picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            except Exception as e:
                logger.warning("Could not enable continuous autofocus: %s", e)
        except Exception as err:
            raise RuntimeError(f"Failed to start Picamera2: {err}") from err

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Capture and return a single BGR frame from the camera."""
        if self.source == "webcam":
            if self._cap is None:
                return False, None
            ret, frame = self._cap.read()
            return ret, frame
        elif self.source == "pi":
            if self._picam2 is None:
                return False, None
            try:
                # Picamera2 returns RGBA/XBGR, convert to standard BGR
                frame = self._picam2.capture_array()
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                return True, frame
            except Exception as err:
                logger.error("Error capturing Picamera2 frame: %s", err)
                return False, None
        return False, None

    def release(self) -> None:
        """Release camera resources cleanly."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._picam2 is not None:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception:
                pass
            self._picam2 = None

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


# -----------------------------------------------------------------------------
# ONNX Inference Predictor
# -----------------------------------------------------------------------------
class Predictor:
    """Lightweight, fast CPU ONNX inference engine using OpenCV DNN.

    Loads the exported MobileNetV2 ONNX model and metadata, preprocesses input crops,
    and returns calibrated softmax probabilities for ["fresh", "rotten"].
    """

    def __init__(
        self,
        onnx_path: Path = MODEL_ONNX_PATH,
        metadata_path: Path = METADATA_PATH,
    ):
        self.onnx_path = Path(onnx_path)
        self.metadata_path = Path(metadata_path)

        if not self.onnx_path.exists():
            raise FileNotFoundError(
                f"No trained model found at '{self.onnx_path}'.\n"
                "Problem: Missing trained ONNX model artifact.\n"
                "Cause: Model has not been trained or exported yet.\n"
                "Fix: Place images in data/ and run: python train.py --epochs 15 --batch-size 16"
            )

        self.metadata = load_metadata(self.metadata_path)
        self.classes = self.metadata["classes"]
        self.size = int(self.metadata["size"])
        self.default_threshold = float(self.metadata.get("threshold", DEFAULT_THRESHOLD))

        try:
            self.net = cv2.dnn.readNetFromONNX(str(self.onnx_path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception as err:
            raise RuntimeError(
                f"Failed to load ONNX model via OpenCV DNN from '{self.onnx_path}': {err}\n"
                "Fix: Re-export the model using 'python export_model.py'."
            ) from err

    def predict(self, bgr_crop: np.ndarray) -> np.ndarray:
        """Run inference on a cropped BGR image and return softmax probabilities.

        Args:
            bgr_crop: BGR square crop from camera or file.

        Returns:
            1D numpy float32 array with probabilities for [fresh_score, rotten_score].
        """
        tensor = tensor_image(bgr_crop)

        self.net.setInput(tensor)
        logits = self.net.forward()

        if logits is None or logits.size != len(self.classes):
            raise ValueError(
                f"Unexpected model output shape: {getattr(logits, 'shape', None)}. "
                f"Expected 2 class logits for {self.classes}."
            )

        # Numerically stable softmax computation
        logits_flat = logits.flatten().astype(np.float64)
        shift_logits = logits_flat - np.max(logits_flat)
        exps = np.exp(shift_logits)
        probs = exps / np.sum(exps)

        # Return as float32 array [p_fresh, p_rotten]
        return probs.astype(np.float32)


# -----------------------------------------------------------------------------
# CSV Logging Helper
# -----------------------------------------------------------------------------
def log_scan_event(
    batch_id: str,
    result: str,
    fresh_score: float,
    rotten_score: float,
    threshold: float,
    inference_ms: float,
    log_path: Path = SCAN_LOG_PATH,
) -> None:
    """Record a single scan event to the CSV log file.

    Args:
        batch_id: Unique string identifier for the active scan batch.
        result: Classification outcome ('FRESH', 'ROTTEN', or 'UNCERTAIN').
        fresh_score: Probability score for the 'fresh' class (0.0 - 1.0).
        rotten_score: Probability score for the 'rotten' class (0.0 - 1.0).
        threshold: Confidence threshold used for certainty determination.
        inference_ms: Inference execution time in milliseconds.
        log_path: Target CSV log filepath.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()

    timestamp = datetime.now().astimezone().isoformat()

    row = [
        timestamp,
        batch_id,
        result.upper(),
        f"{fresh_score:.4f}",
        f"{rotten_score:.4f}",
        f"{threshold:.2f}",
        f"{inference_ms:.1f}",
    ]

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "time",
                    "batch",
                    "result",
                    "fresh_score",
                    "rotten_score",
                    "threshold",
                    "inference_ms",
                ]
            )
        writer.writerow(row)
