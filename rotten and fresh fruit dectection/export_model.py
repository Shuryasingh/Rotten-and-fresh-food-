"""Model export and verification tool.

Exports trained PyTorch MobileNetV2 checkpoint to an optimized ONNX model artifact,
creates the self-describing metadata.json, and executes strict numerical verification
comparing PyTorch output against OpenCV DNN ONNX inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from common import (
    CLASSES,
    DEFAULT_SUPPORTED_FRUIT,
    DEFAULT_THRESHOLD,
    METADATA_PATH,
    MODEL_ONNX_PATH,
    MODEL_PT_PATH,
    SIZE,
    save_metadata,
)


def build_mobilenet_model(num_classes: int = 2) -> nn.Module:
    """Build MobileNetV2 architecture with custom 2-class head."""
    model = mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )
    return model


def export_and_verify(
    checkpoint_path: Path = MODEL_PT_PATH,
    onnx_path: Path = MODEL_ONNX_PATH,
    metadata_path: Path = METADATA_PATH,
    threshold: float = DEFAULT_THRESHOLD,
    fruit_name: str = DEFAULT_SUPPORTED_FRUIT,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> bool:
    """Export PyTorch model to ONNX and verify numerical parity with OpenCV DNN."""
    print("=" * 70)
    print("FRUIT FRESHNESS DETECTOR - MODEL EXPORT & ONNX VERIFICATION")
    print("=" * 70)
    print(f"Checkpoint source: {checkpoint_path.resolve()}")
    print(f"ONNX destination:  {onnx_path.resolve()}")
    print(f"Metadata file:     {metadata_path.resolve()}\n")

    if not checkpoint_path.exists():
        print(
            f"[-] ERROR: PyTorch checkpoint not found at '{checkpoint_path}'.\n"
            "Problem: No trained PyTorch weights found.\n"
            "Fix: Train the model first using: python train.py --epochs 15 --batch-size 16"
        )
        return False

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load PyTorch model
    print("[*] Loading PyTorch model architecture and weights...")
    model = build_mobilenet_model(num_classes=len(CLASSES))
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    except Exception as err:
        print(f"[-] Failed to load checkpoint: {err}")
        return False

    model.eval()

    # 2. Export to ONNX
    print("[*] Exporting model to ONNX format (input shape: [1, 3, 224, 224])...")
    dummy_input = torch.randn(1, 3, SIZE, SIZE, dtype=torch.float32)

    try:
        # PyTorch 2.x supports dynamo=False to use the robust TorchScript ONNX exporter
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=None,  # Fixed shape optimal for OpenCV DNN deployment
                dynamo=False,
            )
        except TypeError:
            # Fallback for earlier PyTorch versions where 'dynamo' argument is not present
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=None,
            )
        print(f"[+] ONNX model successfully written to {onnx_path.name}")
    except Exception as err:
        print(f"[-] PyTorch ONNX export failed: {err}")
        return False

    # 3. Create metadata.json
    print("[*] Writing model metadata.json...")
    metadata = {
        "classes": CLASSES,
        "size": SIZE,
        "model": "MobileNetV2",
        "format": "ONNX",
        "created_at": datetime.now().astimezone().isoformat(),
        "threshold": threshold,
        "supported_fruit": fruit_name,
        "framework": f"PyTorch {torch.__version__} / Torchvision",
        "input_shape": [1, 3, SIZE, SIZE],
        "input_name": "input",
        "output_name": "output",
    }
    save_metadata(metadata, metadata_path)
    print(f"[+] Metadata written to {metadata_path.name}")

    # 4. Mandatory Verification: PyTorch vs OpenCV DNN
    print("\n[*] RUNNING MANDATORY PYTORCH vs ONNX VERIFICATION...")
    try:
        net = cv2.dnn.readNetFromONNX(str(onnx_path))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception as err:
        print(f"[-] FAILED to load exported ONNX in OpenCV DNN: {err}")
        return False

    # Test across multiple pseudo-random and edge test tensors
    verification_passed = True
    test_seeds = [42, 123, 999, 2026]

    for seed in test_seeds:
        torch.manual_seed(seed)
        test_tensor = torch.randn(1, 3, SIZE, SIZE, dtype=torch.float32)
        test_np = test_tensor.numpy()

        # PyTorch forward
        with torch.no_grad():
            pt_logits = model(test_tensor).numpy()
            pt_shift = pt_logits - np.max(pt_logits, axis=-1, keepdims=True)
            pt_probs = (np.exp(pt_shift) / np.sum(np.exp(pt_shift), axis=-1, keepdims=True)).flatten()

        # OpenCV DNN forward
        net.setInput(test_np)
        cv_logits = net.forward()
        cv_shift = cv_logits - np.max(cv_logits, axis=-1, keepdims=True)
        cv_probs = (np.exp(cv_shift) / np.sum(np.exp(cv_shift), axis=-1, keepdims=True)).flatten()

        # Compare outputs
        max_diff = float(np.max(np.abs(pt_probs - cv_probs)))
        close = np.allclose(pt_probs, cv_probs, atol=atol, rtol=rtol)

        if not close:
            print(f"[-] Output mismatch on test tensor (seed {seed})!")
            print(f"    PyTorch probabilities: {pt_probs}")
            print(f"    OpenCV probabilities:  {cv_probs}")
            print(f"    Max Absolute Diff:     {max_diff:.6f} (tolerance: {atol})")
            verification_passed = False
            break

    if verification_passed:
        print("-" * 70)
        print("[+] Export comparison PASSED: PyTorch and OpenCV DNN match within tolerance.")
        print(f"    Maximum observed probability divergence < {atol}")
        print("    Model is verified and ready for live scanning and Raspberry Pi deployment.")
        print("-" * 70 + "\n")
        return True
    else:
        print("\n[-] VERIFICATION FAILED: Numerical divergence between PyTorch and OpenCV DNN exceeded tolerance.")
        print("    DO NOT deploy this model artifact.\n")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export trained PyTorch checkpoint to ONNX and verify numerical parity."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=MODEL_PT_PATH,
        help="Path to trained PyTorch .pt checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_ONNX_PATH,
        help="Destination path for ONNX model",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=METADATA_PATH,
        help="Destination path for metadata.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Default confidence threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--fruit",
        type=str,
        default=DEFAULT_SUPPORTED_FRUIT,
        help=f"Supported fruit variety (default: {DEFAULT_SUPPORTED_FRUIT})",
    )

    args = parser.parse_args()
    success = export_and_verify(
        checkpoint_path=args.checkpoint,
        onnx_path=args.output,
        metadata_path=args.metadata,
        threshold=args.threshold,
        fruit_name=args.fruit,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
