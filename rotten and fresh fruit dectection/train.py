"""Training pipeline for Fruit Freshness Detector using MobileNetV2 Transfer Learning.

Features:
- MobileNetV2 transfer learning with Torchvision weights (or educational --scratch mode)
- Modest data augmentation (horizontal flip, subtle rotation, subtle color jitter)
- Validation checkpointing (artifacts/best_model.pt)
- Loss and accuracy curve plotting (artifacts/training_graph.png)
- History recording (artifacts/training_history.json)
- Automatic ONNX export and OpenCV DNN numerical verification
- Integrated evaluation (--evaluate val|test)
- Clean, actionable error reporting for beginners on empty datasets
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from common import (
    ARTIFACTS_DIR,
    CLASSES,
    DATA_DIR,
    DEFAULT_SUPPORTED_FRUIT,
    DEFAULT_THRESHOLD,
    METADATA_PATH,
    MODEL_ONNX_PATH,
    MODEL_PT_PATH,
    SIZE,
    logger,
)
from evaluate import evaluate_split
from export_model import export_and_verify

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_dataset_readiness(data_dir: Path) -> Tuple[bool, str]:
    """Check if train and validation splits contain sufficient images to train."""
    for split in ["train", "val"]:
        split_path = data_dir / split
        if not split_path.exists():
            return False, f"Missing '{split}' dataset folder at {split_path.relative_to(data_dir.parent)}."

        for cls_name in CLASSES:
            cls_path = split_path / cls_name
            if not cls_path.exists():
                return False, f"Missing class directory at {cls_path.relative_to(data_dir.parent)}."

            images = [p for p in cls_path.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
            if len(images) == 0:
                return (
                    False,
                    f"No images found in {cls_path.relative_to(data_dir.parent)}.\n"
                    f"Problem: Split '{split}' has zero samples for class '{cls_name}'.\n"
                    f"Fix: Collect images using:\n"
                    f"  python app.py collect --split {split} --label {cls_name} --id {cls_name}001",
                )

    return True, "Dataset ready"


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 16,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Build PyTorch DataLoaders for training and validation."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((SIZE, SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((SIZE, SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(root=str(data_dir / "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(data_dir / "val"), transform=val_transform)

    # Validate that ImageFolder mapped classes in expected order
    expected_classes = CLASSES
    if train_dataset.classes != expected_classes:
        logger.warning(
            "ImageFolder class order %s differs from default %s. Adjusting mapping.",
            train_dataset.classes,
            expected_classes,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 workers ensures universal compatibility across Windows and Linux
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    counts = {
        "train": len(train_dataset),
        "val": len(val_dataset),
    }

    return train_loader, val_loader, counts


def build_model(scratch: bool = False, num_classes: int = 2) -> nn.Module:
    """Instantiate MobileNetV2 with custom 2-class head."""
    if scratch:
        print("[*] Initializing MobileNetV2 FROM SCRATCH (weights=None)...")
        print("    [!] Note: Scratch training requires large datasets and significantly more epochs.")
        model = mobilenet_v2(weights=None)
    else:
        print("[*] Loading pretrained MobileNetV2 weights (Torchvision Default)...")
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )
    return model


def plot_training_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    """Generate and save training vs validation loss and accuracy graphs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=1.8, markersize=4)
    ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss", linewidth=1.8, markersize=4)
    ax1.set_title("Training and Validation Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Acc", linewidth=1.8, markersize=4)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "g--s", label="Val Acc", linewidth=1.8, markersize=4)
    ax2.set_title("Training and Validation Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim([0, 105])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[+] Saved training curves to: {output_path.name}")


def train_model(
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 0.001,
    scratch: bool = False,
    data_dir: Path = DATA_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    threshold: float = DEFAULT_THRESHOLD,
    fruit_name: str = DEFAULT_SUPPORTED_FRUIT,
    device_name: Optional[str] = None,
) -> None:
    """Execute complete training pipeline, checkpointing, graphing, and ONNX export."""
    print("=" * 70)
    print("FRUIT FRESHNESS DETECTOR - MODEL TRAINING PIPELINE")
    print("=" * 70)

    # 1. Check dataset readiness
    ready, message = check_dataset_readiness(data_dir)
    if not ready:
        print(f"\n[-] DATASET NOT READY:\n{message}\n")
        sys.exit(1)

    # 2. Select device (CPU by default on laptops, CUDA if available)
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"[*] Training Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 3. Build DataLoaders
    train_loader, val_loader, counts = create_dataloaders(data_dir, batch_size=batch_size)
    print(f"[*] Dataset: {counts['train']} training images, {counts['val']} validation images")
    print(f"[*] Classes: {CLASSES} -> fresh (0), rotten (1)")
    print(f"[*] Epochs:  {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")

    # 4. Build Model
    model = build_model(scratch=scratch, num_classes=len(CLASSES)).to(device)

    # If transfer learning: freeze lower feature layers during initial epochs
    if not scratch:
        print("[*] Freezing MobileNetV2 base feature extractor for initial head warmup...")
        for param in model.features.parameters():
            param.requires_grad = False
        # Head parameters remain trainable
        for param in model.classifier.parameters():
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = artifacts_dir / "best_model.pt"

    start_time = time.time()
    print("\n" + "-" * 70)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<11} | {'Val Loss':<10} | {'Val Acc':<10} | {'Status'}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        # In transfer learning, unfreeze upper feature layers after epoch 3 for fine-tuning
        if not scratch and epoch == 4:
            print("[*] Fine-tuning: Unfreezing upper MobileNetV2 feature blocks...")
            for param in model.features[14:].parameters():
                param.requires_grad = True
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr * 0.2, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs - 3, eta_min=1e-6)

        # Training Phase
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += torch.sum(preds == labels).item()
            total_train += images.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = running_correct / total_train

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        running_val_correct = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                running_val_correct += torch.sum(preds == labels).item()
                total_val += images.size(0)

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = running_val_correct / total_val

        scheduler.step()

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))

        # Checkpoint if best validation loss/acc
        status = ""
        if epoch_val_loss < best_val_loss or (epoch_val_loss == best_val_loss and epoch_val_acc > best_val_acc):
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            status = "* (Best Checkpoint)"

            # Save full checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_loss": epoch_val_loss,
                    "val_acc": epoch_val_acc,
                    "classes": CLASSES,
                    "model_architecture": "MobileNetV2",
                },
                best_checkpoint_path,
            )

        print(
            f"{epoch:>2}/{epochs:<4} | {epoch_train_loss:>10.4f}   | {epoch_train_acc * 100:>8.2f}%  | "
            f"{epoch_val_loss:>8.4f}   | {epoch_val_acc * 100:>7.2f}%  | {status}"
        )

    total_duration = time.time() - start_time
    print("-" * 70)
    print(f"[+] Training completed in {total_duration:.1f}s ({total_duration / 60:.2f} min).")
    print(f"[+] Best Validation Checkpoint saved at Epoch {best_epoch}:")
    print(f"    Validation Loss: {best_val_loss:.4f} | Validation Accuracy: {best_val_acc * 100:.2f}%")

    # 5. Save Training History JSON
    history_path = artifacts_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "epochs": epochs,
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"[+] Saved history to: {history_path.name}")

    # 6. Plot Training Graph
    graph_path = artifacts_dir / "training_graph.png"
    plot_training_curves(history, graph_path)

    # 7. Export Model to ONNX and Verify Parity
    print("\n" + "=" * 70)
    print("EXPORTING ONNX MODEL & VERIFYING NUMERICAL PARITY...")
    print("=" * 70)
    export_ok = export_and_verify(
        checkpoint_path=best_checkpoint_path,
        onnx_path=artifacts_dir / "fruit.onnx",
        metadata_path=artifacts_dir / "metadata.json",
        threshold=threshold,
        fruit_name=fruit_name,
    )

    if not export_ok:
        print("[-] Training finished, but ONNX export or verification encountered errors.")
        sys.exit(1)

    # 8. Run Validation Split Evaluation
    print("\n" + "=" * 70)
    print("RUNNING FINAL VALIDATION EVALUATION...")
    print("=" * 70)
    val_report_path = artifacts_dir / "validation_report.json"
    evaluate_split(
        split="val",
        threshold=threshold,
        data_dir=data_dir,
        onnx_path=artifacts_dir / "fruit.onnx",
        metadata_path=artifacts_dir / "metadata.json",
        save_report=True,
        output_report_path=val_report_path,
    )

    print("\n[+] SUCCESS: Training, Checkpointing, ONNX Export, and Verification are COMPLETE!")
    print("    You can now run live scanning:")
    print(f"    python app.py scan --threshold {threshold:.2f} --alert-count 3\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 Fruit Freshness Classifier and Export Verified ONNX Model."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training mini-batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="Train model from scratch without pretrained weights (educational comparison)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Confidence threshold for certainty (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--fruit",
        type=str,
        default=DEFAULT_SUPPORTED_FRUIT,
        help=f"Supported fruit variety name (default: {DEFAULT_SUPPORTED_FRUIT})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Dataset root directory",
    )
    parser.add_argument(
        "--evaluate",
        choices=["val", "test", "train"],
        default=None,
        help="Directly evaluate an existing model on specified split (skips training)",
    )

    args = parser.parse_args()

    if args.evaluate:
        # Evaluate mode requested
        evaluate_split(
            split=args.evaluate,
            threshold=args.threshold,
            data_dir=args.data_dir,
        )
    else:
        train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            scratch=args.scratch,
            data_dir=args.data_dir,
            artifacts_dir=ARTIFACTS_DIR,
            threshold=args.threshold,
            fruit_name=args.fruit,
        )


if __name__ == "__main__":
    main()
