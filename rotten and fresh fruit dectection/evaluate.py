"""Evaluation module for Fruit Freshness Detector.

Computes comprehensive classification metrics on validation or test splits:
- 2x2 Forced-choice Confusion Matrix (Rows = True, Cols = Predicted)
- Thresholded Confusion Matrix with UNCERTAIN category
- Overall Forced-choice Accuracy
- Rotten Recall (Critical safety metric: % of rotten specimens identified)
- Fresh Recall & Precision
- Accepted Fraction & Accepted Accuracy (confidence >= threshold)
- List of individual misclassifications

Important: Visual appearance classifier only — not a food-safety certification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from common import (
    CLASSES,
    DATA_DIR,
    DEFAULT_THRESHOLD,
    METADATA_PATH,
    MODEL_ONNX_PATH,
    REPORTS_DIR,
    Predictor,
    logger,
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def evaluate_split(
    split: str = "val",
    threshold: float = DEFAULT_THRESHOLD,
    data_dir: Path = DATA_DIR,
    onnx_path: Path = MODEL_ONNX_PATH,
    metadata_path: Path = METADATA_PATH,
    save_report: bool = True,
    output_report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run full evaluation on a specific dataset split.

    Args:
        split: 'val' or 'test'.
        threshold: Confidence threshold for confident classification.
        data_dir: Dataset root folder.
        onnx_path: Path to exported ONNX model.
        metadata_path: Path to metadata.json.
        save_report: Whether to write JSON report to disk.
        output_report_path: Optional explicit output report path.

    Returns:
        Dictionary containing all evaluation metrics.
    """
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    # Check images exist in split
    image_paths: List[Tuple[Path, str, int]] = []
    for cls_name in CLASSES:
        cls_dir = split_dir / cls_name
        if cls_dir.exists():
            for p in cls_dir.iterdir():
                if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
                    image_paths.append((p, cls_name, CLASSES.index(cls_name)))

    if not image_paths:
        print(
            f"\n[-] EVALUATION ERROR: No images found in '{split_dir.relative_to(data_dir.parent)}'.\n"
            f"Fix: Place test/validation images in {split_dir.relative_to(data_dir.parent)}/fresh and "
            f"{split_dir.relative_to(data_dir.parent)}/rotten before running evaluation."
        )
        sys.exit(1)

    # Initialize predictor
    predictor = Predictor(onnx_path=onnx_path, metadata_path=metadata_path)

    # Confusion Matrix tracking
    # Forced choice: 2x2 [True][Pred]
    cm_forced = np.zeros((2, 2), dtype=int)
    # Thresholded: 2x3 [True][Fresh, Rotten, Uncertain]
    cm_thresholded = np.zeros((2, 3), dtype=int)

    total_samples = len(image_paths)
    class_counts = {c: 0 for c in CLASSES}
    forced_errors: List[Dict[str, Any]] = []
    uncertain_samples: List[Dict[str, Any]] = []

    accepted_count = 0
    accepted_correct = 0

    print("=" * 70)
    print(f"FRUIT FRESHNESS DETECTOR - EVALUATION REPORT ({split.upper()} SPLIT)")
    print("=" * 70)
    print(f"Dataset Split:        {split}")
    print(f"Total Images:         {total_samples}")
    print(f"Confidence Threshold: {threshold * 100:.1f}%")
    print(f"Model Artifact:       {onnx_path.name}")
    print("-" * 70)

    for img_path, true_label, true_idx in image_paths:
        class_counts[true_label] += 1
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        probs = predictor.predict(bgr)
        fresh_score = float(probs[0])
        rotten_score = float(probs[1])

        winner_idx = int(np.argmax(probs))
        confidence = float(probs[winner_idx])
        pred_label_forced = CLASSES[winner_idx]

        # Update forced-choice confusion matrix
        cm_forced[true_idx, winner_idx] += 1

        is_forced_correct = (winner_idx == true_idx)
        if not is_forced_correct:
            forced_errors.append(
                {
                    "filename": img_path.name,
                    "true_class": true_label,
                    "predicted_forced": pred_label_forced,
                    "confidence": round(confidence, 4),
                    "fresh_score": round(fresh_score, 4),
                    "rotten_score": round(rotten_score, 4),
                }
            )

        # Thresholded logic
        if confidence >= threshold:
            pred_col = winner_idx  # 0: fresh, 1: rotten
            cm_thresholded[true_idx, pred_col] += 1
            accepted_count += 1
            if is_forced_correct:
                accepted_correct += 1
        else:
            cm_thresholded[true_idx, 2] += 1  # 2: uncertain
            uncertain_samples.append(
                {
                    "filename": img_path.name,
                    "true_class": true_label,
                    "confidence": round(confidence, 4),
                    "fresh_score": round(fresh_score, 4),
                    "rotten_score": round(rotten_score, 4),
                }
            )

    # -------------------------------------------------------------------------
    # Metric Calculations
    # -------------------------------------------------------------------------
    # Forced-choice accuracy
    total_forced_correct = int(np.trace(cm_forced))
    forced_accuracy = (total_forced_correct / total_samples) if total_samples > 0 else 0.0

    # Rotten Metrics (Class 1 = Rotten)
    # TP = cm_forced[1, 1], FN = cm_forced[1, 0], FP = cm_forced[0, 1], TN = cm_forced[0, 0]
    tp_rotten = int(cm_forced[1, 1])
    fn_rotten = int(cm_forced[1, 0])
    fp_rotten = int(cm_forced[0, 1])
    rotten_support = class_counts["rotten"]

    rotten_recall = (tp_rotten / (tp_rotten + fn_rotten)) if (tp_rotten + fn_rotten) > 0 else 0.0
    rotten_precision = (tp_rotten / (tp_rotten + fp_rotten)) if (tp_rotten + fp_rotten) > 0 else 0.0

    # Fresh Metrics (Class 0 = Fresh)
    tp_fresh = int(cm_forced[0, 0])
    fn_fresh = int(cm_forced[0, 1])
    fp_fresh = int(cm_forced[1, 0])
    fresh_support = class_counts["fresh"]

    fresh_recall = (tp_fresh / (tp_fresh + fn_fresh)) if (tp_fresh + fn_fresh) > 0 else 0.0
    fresh_precision = (tp_fresh / (tp_fresh + fp_fresh)) if (tp_fresh + fp_fresh) > 0 else 0.0

    # Thresholded Metrics
    accepted_fraction = (accepted_count / total_samples) if total_samples > 0 else 0.0
    accepted_accuracy = (accepted_correct / accepted_count) if accepted_count > 0 else 0.0
    uncertain_count = total_samples - accepted_count
    uncertain_fraction = (uncertain_count / total_samples) if total_samples > 0 else 0.0

    # -------------------------------------------------------------------------
    # Console Output
    # -------------------------------------------------------------------------
    print("\n1. FORCED-CHOICE CONFUSION MATRIX (2x2):")
    print("   [Rows = True Condition | Columns = Predicted Condition]")
    print(f"                 Pred Fresh   Pred Rotten   Total")
    print(f"   True Fresh   {cm_forced[0, 0]:>10}   {cm_forced[0, 1]:>11}   {fresh_support:>5}")
    print(f"   True Rotten  {cm_forced[1, 0]:>10}   {cm_forced[1, 1]:>11}   {rotten_support:>5}")
    print(f"   Total        {cm_forced[:, 0].sum():>10}   {cm_forced[:, 1].sum():>11}   {total_samples:>5}")

    print("\n2. THRESHOLDED CONFUSION MATRIX:")
    print(f"   (Certainty Threshold = {threshold * 100:.0f}%)")
    print(f"                 Pred Fresh   Pred Rotten   Uncertain   Total")
    print(f"   True Fresh   {cm_thresholded[0, 0]:>10}   {cm_thresholded[0, 1]:>11}   {cm_thresholded[0, 2]:>9}   {fresh_support:>5}")
    print(f"   True Rotten  {cm_thresholded[1, 0]:>10}   {cm_thresholded[1, 1]:>11}   {cm_thresholded[1, 2]:>9}   {rotten_support:>5}")

    print("\n3. PERFORMANCE METRICS:")
    print(f"   - Forced-Choice Accuracy: {forced_accuracy * 100:.2f}% ({total_forced_correct}/{total_samples})")
    print(f"   - Rotten Recall:          {rotten_recall * 100:.2f}% ({tp_rotten}/{rotten_support})  [CRITICAL METRIC]")
    print(f"   - Rotten Precision:       {rotten_precision * 100:.2f}%")
    print(f"   - Fresh Recall:           {fresh_recall * 100:.2f}% ({tp_fresh}/{fresh_support})")
    print(f"   - Fresh Precision:        {fresh_precision * 100:.2f}%")
    print(f"   - Accepted Fraction:      {accepted_fraction * 100:.2f}% ({accepted_count}/{total_samples} confident)")
    print(f"   - Accepted Accuracy:      {accepted_accuracy * 100:.2f}% ({accepted_correct}/{accepted_count} correct)")
    print(f"   - Uncertain / Rejected:   {uncertain_fraction * 100:.2f}% ({uncertain_count}/{total_samples})")

    if forced_errors:
        print(f"\n4. FORCED-CHOICE MISCLASSIFICATIONS ({len(forced_errors)}):")
        for err in forced_errors[:8]:
            print(
                f"   - {err['filename']}: True={err['true_class']}, Pred={err['predicted_forced']} "
                f"(Score={err['confidence'] * 100:.1f}%)"
            )
        if len(forced_errors) > 8:
            print(f"   ... and {len(forced_errors) - 8} more.")
    else:
        print("\n4. MISCLASSIFICATIONS: 0 errors on this split!")

    print("\n" + "=" * 70)
    print("NOTE: Visual appearance classifier only — not a food-safety determination.")
    print("=" * 70 + "\n")

    report_data = {
        "split": split,
        "total_samples": total_samples,
        "class_counts": class_counts,
        "threshold": threshold,
        "confusion_matrix_forced_rows_true_cols_pred": {
            "fresh": {"pred_fresh": int(cm_forced[0, 0]), "pred_rotten": int(cm_forced[0, 1])},
            "rotten": {"pred_fresh": int(cm_forced[1, 0]), "pred_rotten": int(cm_forced[1, 1])},
        },
        "confusion_matrix_thresholded_rows_true_cols_pred": {
            "fresh": {
                "pred_fresh": int(cm_thresholded[0, 0]),
                "pred_rotten": int(cm_thresholded[0, 1]),
                "uncertain": int(cm_thresholded[0, 2]),
            },
            "rotten": {
                "pred_fresh": int(cm_thresholded[1, 0]),
                "pred_rotten": int(cm_thresholded[1, 1]),
                "uncertain": int(cm_thresholded[1, 2]),
            },
        },
        "metrics": {
            "forced_accuracy": round(forced_accuracy, 4),
            "rotten_recall": round(rotten_recall, 4),
            "rotten_precision": round(rotten_precision, 4),
            "fresh_recall": round(fresh_recall, 4),
            "fresh_precision": round(fresh_precision, 4),
            "accepted_fraction": round(accepted_fraction, 4),
            "accepted_accuracy": round(accepted_accuracy, 4),
            "uncertain_count": uncertain_count,
        },
        "misclassifications": forced_errors,
        "uncertain_samples": uncertain_samples,
        "safety_disclaimer": "Visual appearance classifier only — not a food-safety determination.",
    }

    if save_report:
        if output_report_path is None:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            output_report_path = REPORTS_DIR / f"evaluation_{split}.json"
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"[+] Evaluation report saved to: {output_report_path.resolve()}")

    return report_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained Fruit Freshness ONNX model on validation or test splits."
    )
    parser.add_argument(
        "--split",
        choices=["val", "test", "train"],
        default="val",
        help="Dataset split to evaluate (default: val)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Certainty threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Dataset root directory",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=MODEL_ONNX_PATH,
        help="Path to fruit.onnx model artifact",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=METADATA_PATH,
        help="Path to metadata.json artifact",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Explicit output JSON report file path",
    )

    args = parser.parse_args()
    evaluate_split(
        split=args.split,
        threshold=args.threshold,
        data_dir=args.data_dir,
        onnx_path=args.onnx,
        metadata_path=args.metadata,
        save_report=True,
        output_report_path=args.out,
    )


if __name__ == "__main__":
    main()
