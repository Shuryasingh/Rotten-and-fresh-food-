"""Main application for Fruit Freshness Detector.

Supports two primary execution modes:
1. 'collect': Interactive dataset collection with physical specimen isolation enforcement.
2. 'scan': Live real-time inspection, inference, thresholding, batch tracking, and CSV logging.
"""

from __future__ import annotations

import argparse
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from common import (
    CLASSES,
    DATA_DIR,
    DEFAULT_ALERT_COUNT,
    DEFAULT_SUPPORTED_FRUIT,
    DEFAULT_THRESHOLD,
    FRUIT_ID_REGEX,
    Camera,
    Predictor,
    log_scan_event,
    square,
)


# -----------------------------------------------------------------------------
# Specimen Isolation Verification Helper
# -----------------------------------------------------------------------------
def verify_specimen_isolation(target_split: str, fruit_id: str, data_dir: Path = DATA_DIR) -> Optional[str]:
    """Verify that a physical fruit identity does not already exist in a different split.

    Returns:
        Name of conflicting split if a violation is found, or None if safe.
    """
    for split in ["train", "val", "test"]:
        if split == target_split:
            continue
        for cls_name in CLASSES:
            folder = data_dir / split / cls_name
            if not folder.exists():
                continue
            for img_path in folder.glob(f"{fruit_id}__*"):
                if img_path.is_file():
                    return split
    return None


# -----------------------------------------------------------------------------
# Mode 1: Interactive Dataset Collection
# -----------------------------------------------------------------------------
def run_collection(
    split: str,
    label: str,
    fruit_id: str,
    source: str = "webcam",
    index: int = 0,
    data_dir: Path = DATA_DIR,
) -> None:
    """Run interactive camera tool to capture training/val/test crops for a specific fruit specimen."""
    # 1. Validate fruit identity string
    if not FRUIT_ID_REGEX.match(fruit_id):
        print(
            f"[-] INVALID FRUIT ID: '{fruit_id}'\n"
            "Problem: Fruit ID contains invalid characters or spaces.\n"
            "Fix: Use only letters, digits, and hyphens (e.g., 'apple001', 'specimen-042')."
        )
        sys.exit(1)

    # 2. Check specimen isolation across splits
    conflict_split = verify_specimen_isolation(split, fruit_id, data_dir)
    if conflict_split:
        print(
            f"\n[-] CRITICAL DATASET ISOLATION ERROR:\n"
            f"    Physical fruit specimen ID '{fruit_id}' is already registered in '{conflict_split}' split!\n"
            f"    Rule: A single physical fruit specimen MUST belong to exactly ONE dataset split to prevent data leakage.\n"
            f"    Fix: If this is a new physical fruit, choose a new fruit ID (e.g., '{fruit_id}-b' or '{fruit_id}2')."
        )
        sys.exit(1)

    # 3. Ensure target directory exists
    target_dir = data_dir / split / label
    target_dir.mkdir(parents=True, exist_ok=True)

    # Count existing images for this specimen in target folder
    existing_for_id = len(list(target_dir.glob(f"{fruit_id}__*")))

    print("=" * 65)
    print("FRUIT FRESHNESS DETECTOR - DATASET COLLECTION MODE")
    print("=" * 65)
    print(f"Split:       {split}")
    print(f"Class:       {label}")
    print(f"Fruit ID:    {fruit_id}")
    print(f"Target dir:  {target_dir.resolve()}")
    print(f"Existing:    {existing_for_id} images saved for this fruit ID")
    print("\nInstructions:")
    print("  - Place ONE fruit inside the green inspection square.")
    print("  - Press 'S' to capture and save the square crop.")
    print("  - Rotate or flip the fruit between shots for multiple angles.")
    print("  - Press 'Q' or 'ESC' when finished with this fruit specimen.\n")

    try:
        camera = Camera(source=source, index=index)
    except Exception as err:
        print(f"[-] CAMERA ERROR: {err}")
        sys.exit(1)

    window_name = f"Data Collection - [{split.upper()} | {label.upper()} | ID: {fruit_id}]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    saved_count = existing_for_id
    last_save_time = 0.0
    status_msg = f"Ready. Captured: {saved_count}"
    status_time = 0.0

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("[-] Failed to read frame from camera.")
                break

            h, w = frame.shape[:2]
            crop, (x, y, side) = square(frame)

            display = frame.copy()

            # Top info bar
            cv2.rectangle(display, (0, 0), (w, 60), (25, 25, 25), -1)
            cv2.putText(
                display,
                f"COLLECTION: [{split.upper()}] -> [{label.upper()}]",
                (15, 25),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Fruit ID: {fruit_id}  |  Total Saved for ID: {saved_count}",
                (15, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 255, 200),
                1,
                cv2.LINE_AA,
            )

            # Draw green inspection box
            cv2.rectangle(display, (x, y), (x + side, y + side), (0, 255, 0), 2)
            cv2.putText(
                display,
                "Crop Area (Saved to Disk)",
                (x + 5, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            # Flash status message if recent
            if time.time() - status_time < 1.5:
                cv2.rectangle(display, (x, y + side + 10), (x + side, y + side + 35), (0, 180, 0), -1)
                cv2.putText(
                    display,
                    status_msg,
                    (x + 10, y + side + 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Bottom instructions bar
            cv2.rectangle(display, (0, h - 35), (w, h), (25, 25, 25), -1)
            cv2.putText(
                display,
                "S = Save Crop    |    Q / ESC = Quit Session",
                (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("s"), ord("S")):
                # Cooldown check: 300ms
                now = time.time()
                if now - last_save_time >= 0.3:
                    last_save_time = now
                    # Generate unique filename with random hex to avoid collisions
                    unique_token = secrets.token_hex(4)
                    filename = f"{fruit_id}__{unique_token}.jpg"
                    filepath = target_dir / filename

                    # Save ONLY the square cropped image
                    success = cv2.imwrite(str(filepath), crop)
                    if success:
                        saved_count += 1
                        status_msg = f"SAVED #{saved_count}: {filename}"
                        status_time = now
                        print(f"[+] Saved crop: {filepath.relative_to(data_dir.parent)}")
                    else:
                        print(f"[-] Failed to write image: {filepath}")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f"\n[+] Collection session ended. Total images for '{fruit_id}': {saved_count}")


# -----------------------------------------------------------------------------
# Setup & Guidance Helpers
# -----------------------------------------------------------------------------
def print_model_missing_guidance(error_message: str | Exception) -> None:
    """Print clean, actionable setup guidance when model artifact or metadata is missing."""
    print("\n" + "=" * 72)
    print("  [!] MODEL NOT FOUND - QUICK SETUP GUIDE")
    print("=" * 72)
    print("The Fruit Freshness Detector could not find a trained model artifact.")
    print(f"Error Details: {error_message}")
    print("\nTo train the model and generate 'artifacts/fruit.onnx':")
    print("  1. Download public labeled dataset:")
    print("     python download_dataset.py --variety apple")
    print("     (or collect custom specimens: python app.py collect --split train --label fresh --id apple001)")
    print("\n  2. Audit dataset hygiene and isolation:")
    print("     python audit_dataset.py")
    print("\n  3. Train MobileNetV2 and export verified ONNX model:")
    print("     python train.py --epochs 15 --batch-size 16")
    print("\n  4. Re-launch the scanner:")
    print("     python app.py scan")
    print("=" * 72 + "\n")


# -----------------------------------------------------------------------------
# Mode 2: Live Inference Scanner
# -----------------------------------------------------------------------------
def run_scanner(
    threshold: float = DEFAULT_THRESHOLD,
    alert_count: int = DEFAULT_ALERT_COUNT,
    fruit_name: str = DEFAULT_SUPPORTED_FRUIT,
    auto_interval: float = 0.5,
    source: str = "webcam",
    index: int = 0,
    enable_gpio: bool = False,
) -> None:
    """Run real-time presentation-grade fruit freshness inspection and classification application."""
    # 1. Validate GPIO configuration
    led_controller = None
    if enable_gpio:
        if source != "pi":
            print(
                "[-] CONFIGURATION ERROR:\n"
                "Problem: '--gpio' is enabled with '--source webcam'.\n"
                "Cause: GPIO hardware is only available when running on Raspberry Pi.\n"
                "Fix: Remove '--gpio' when running on PC/laptop, or use '--source pi' on Raspberry Pi."
            )
            sys.exit(1)
        try:
            from pi.gpio_config import LedController
            led_controller = LedController()
            print("[+] GPIO LEDs initialized (Green, Yellow, Red).")
        except Exception as err:
            print(f"[-] Failed to initialize GPIO LEDs: {err}")
            sys.exit(1)

    # 2. Initialize Predictor & Model (Graceful missing model handling)
    print("=" * 65)
    print("FRUIT FRESHNESS DETECTOR - LIVE SCANNER")
    print("=" * 65)
    print(f"Supported Fruit:      {fruit_name.upper()}")
    print(f"Confidence Threshold: {threshold * 100:.0f}%")
    print(f"Rotten Alert Count:   {alert_count}")
    print(f"Auto-Scan Interval:   {auto_interval:.2f}s" if auto_interval > 0 else "Auto-Scan Interval:   Every frame")
    print(f"Camera Source:        {source}[{index}]")
    print("-" * 65)

    try:
        predictor = Predictor()
    except (FileNotFoundError, RuntimeError) as err:
        print_model_missing_guidance(err)
        if led_controller:
            led_controller.cleanup()
        return
    except Exception as err:
        print(f"\n[-] UNEXPECTED MODEL INITIALIZATION ERROR:\n{err}\n")
        if led_controller:
            led_controller.cleanup()
        return

    # 3. Initialize Camera
    try:
        camera = Camera(source=source, index=index)
    except Exception as err:
        print(f"\n[-] CAMERA INITIALIZATION ERROR:\n{err}\n")
        if led_controller:
            led_controller.cleanup()
        return

    # 4. Batch State Tracking (Deliberate scan counters)
    def create_batch_id() -> str:
        return f"BATCH-{secrets.token_hex(3).upper()}"

    batch_id = create_batch_id()
    counts = {"FRESH": 0, "ROTTEN": 0, "UNCERTAIN": 0}

    # Deliberate logged scan state
    last_result: Optional[str] = None
    last_score: float = 0.0
    last_fresh_p: float = 0.0
    last_rotten_p: float = 0.0
    last_inference_ms: float = 0.0
    last_scan_time = 0.0

    # Automatic live classification state (separate from batch counts!)
    last_auto_scan_time = 0.0
    live_verdict: Optional[str] = None
    live_score: float = 0.0
    live_fresh_p: float = 0.0
    live_rotten_p: float = 0.0
    live_inference_ms: float = 0.0

    # UI Flash feedback state
    flash_msg: Optional[str] = None
    flash_time = 0.0

    window_name = "Fruit Freshness Detector - Presentation Inspection Console"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"\n[+] System ready! Batch ID: {batch_id}")
    print("Controls in preview window:")
    print("  'S' -> Log and count fruit currently in square (Deliberate capture)")
    print("  'B' -> Start new batch (resets batch counters)")
    print("  'Q' -> Exit scanner cleanly\n")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("[-] Camera frame capture failure.")
                break

            h, w = frame.shape[:2]
            crop, (x, y, side) = square(frame)

            # -----------------------------------------------------------------
            # Automatic Live Classification (Independent of Batch Logging)
            # -----------------------------------------------------------------
            now_perf = time.perf_counter()
            if auto_interval >= 0 and (now_perf - last_auto_scan_time >= auto_interval):
                t0 = time.perf_counter()
                try:
                    probs = predictor.predict(crop)
                    live_inference_ms = (time.perf_counter() - t0) * 1000.0
                    live_fresh_p = float(probs[0])
                    live_rotten_p = float(probs[1])

                    winner_idx = int(np.argmax(probs))
                    confidence = float(probs[winner_idx])
                    winning_class = CLASSES[winner_idx].upper()

                    if confidence >= threshold:
                        live_verdict = winning_class
                    else:
                        live_verdict = "UNCERTAIN"
                    live_score = confidence
                    last_auto_scan_time = now_perf
                except Exception as e:
                    live_verdict = None

            display = frame.copy()

            # -----------------------------------------------------------------
            # UI Layout: Header, Sidebars, Inspection Square, Result Cards
            # -----------------------------------------------------------------
            # Top Banner
            cv2.rectangle(display, (0, 0), (w, 55), (15, 15, 15), -1)
            cv2.putText(
                display,
                "FRUIT FRESHNESS DETECTOR",
                (15, 25),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                (0, 255, 220),
                1,
                cv2.LINE_AA,
            )
            interval_str = f"Auto: {auto_interval:.1f}s" if auto_interval > 0 else "Auto: Realtime"
            cv2.putText(
                display,
                f"SUPPORTED: {fruit_name.upper()}  |  {interval_str}",
                (15, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Threshold: {threshold * 100:.0f}%",
                (w - 150, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # Left Info Box: Batch and Counts
            box_w = 175
            box_h = 145
            box_y = 65
            cv2.rectangle(display, (10, box_y), (10 + box_w, box_y + box_h), (25, 25, 25), -1)
            cv2.rectangle(display, (10, box_y), (10 + box_w, box_y + box_h), (60, 60, 60), 1)

            cv2.putText(
                display,
                f"Batch: {batch_id}",
                (18, box_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Fresh:     {counts['FRESH']}",
                (18, box_y + 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Rotten:    {counts['ROTTEN']}",
                (18, box_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (50, 50, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Uncertain: {counts['UNCERTAIN']}",
                (18, box_y + 94),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Total Logged: {sum(counts.values())}",
                (18, box_y + 118),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                "(Press 'S' to Log Item)",
                (18, box_y + 137),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (140, 140, 140),
                1,
                cv2.LINE_AA,
            )

            # Determine square inspection box color based on LIVE prediction
            if live_verdict == "FRESH":
                box_color = (0, 255, 0)       # Green
            elif live_verdict == "ROTTEN":
                box_color = (0, 0, 255)       # Red
            elif live_verdict == "UNCERTAIN":
                box_color = (0, 215, 255)     # Amber/Yellow
            else:
                box_color = (200, 200, 200)   # Neutral Gray

            # Draw square inspection area
            cv2.rectangle(display, (x, y), (x + side, y + side), box_color, 2)

            # Live classification pill banner above inspection square
            if live_verdict is not None:
                pill_text = f"LIVE: {live_verdict} ({live_score * 100:.1f}%)"
                pill_w = max(220, side)
                cv2.rectangle(display, (x, y - 28), (x + pill_w, y - 2), (25, 25, 25), -1)
                cv2.rectangle(display, (x, y - 28), (x + pill_w, y - 2), box_color, 1)
                cv2.putText(
                    display,
                    pill_text,
                    (x + 8, y - 9),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.55,
                    box_color,
                    1,
                    cv2.LINE_AA,
                )
                # Subtitle below square showing live probabilities and latency
                stats_str = f"Fresh: {live_fresh_p * 100:.0f}% | Rotten: {live_rotten_p * 100:.0f}% | {live_inference_ms:.1f}ms"
                cv2.rectangle(display, (x, y + side + 5), (x + pill_w, y + side + 25), (25, 25, 25), -1)
                cv2.putText(
                    display,
                    stats_str,
                    (x + 8, y + side + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    display,
                    "Place ONE fruit inside square",
                    (x + 5, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Right Card: Last Commited/Logged Scan Result
            card_w = 210
            card_h = 135
            card_x = w - card_w - 10
            card_y = 65

            cv2.rectangle(display, (card_x, card_y), (card_x + card_w, card_y + card_h), (25, 25, 25), -1)
            card_border_color = (60, 60, 60)
            if last_result == "FRESH":
                card_border_color = (0, 255, 0)
            elif last_result == "ROTTEN":
                card_border_color = (0, 0, 255)
            elif last_result == "UNCERTAIN":
                card_border_color = (0, 215, 255)
            cv2.rectangle(display, (card_x, card_y), (card_x + card_w, card_y + card_h), card_border_color, 2)

            cv2.putText(
                display,
                "LAST LOGGED ITEM ('S')",
                (card_x + 10, card_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            if last_result is not None:
                cv2.putText(
                    display,
                    last_result,
                    (card_x + 10, card_y + 54),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.85,
                    card_border_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    f"Score:     {last_score * 100:.1f}%",
                    (card_x + 10, card_y + 78),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    f"Inference: {last_inference_ms:.1f} ms",
                    (card_x + 10, card_y + 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    "Saved to logs/scans.csv",
                    (card_x + 10, card_y + 122),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    (120, 200, 120),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    display,
                    "None Logged",
                    (card_x + 10, card_y + 54),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    (120, 120, 120),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    "Press 'S' to record item",
                    (card_x + 10, card_y + 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (160, 160, 160),
                    1,
                    cv2.LINE_AA,
                )

            # Flash confirmation banner when 'S' is pressed
            if flash_msg and (time.time() - flash_time < 1.5):
                banner_h = 32
                cv2.rectangle(display, (0, h - 70), (w, h - 70 + banner_h), (0, 140, 0), -1)
                cv2.putText(
                    display,
                    f"[LOGGED] {flash_msg} -> Added to Batch Count and CSV",
                    (20, h - 49),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.50,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            # Warning banner if rotten count threshold reached
            elif counts["ROTTEN"] >= alert_count:
                warn_h = 32
                cv2.rectangle(display, (0, h - 70), (w, h - 70 + warn_h), (0, 0, 180), -1)
                cv2.putText(
                    display,
                    f"WARNING: {counts['ROTTEN']} rotten classifications detected in this batch.",
                    (20, h - 48),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Bottom instructions toolbar
            cv2.rectangle(display, (0, h - 35), (w, h), (15, 15, 15), -1)
            cv2.putText(
                display,
                "S = LOG ITEM (Count & CSV)    |    B = NEW BATCH    |    Q / ESC = QUIT",
                (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 220),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("[*] Operator requested application exit.")
                break
            elif key in (ord("b"), ord("B")):
                # Reset Batch
                batch_id = create_batch_id()
                counts = {"FRESH": 0, "ROTTEN": 0, "UNCERTAIN": 0}
                last_result = None
                if led_controller:
                    led_controller.all_off()
                print(f"[+] Started new batch: {batch_id}")
            elif key in (ord("s"), ord("S")):
                # Deliberate capture scan: count item and record to CSV log
                t0 = time.perf_counter()
                probs = predictor.predict(crop)
                inference_ms = (time.perf_counter() - t0) * 1000.0

                fresh_p = float(probs[0])
                rotten_p = float(probs[1])

                winner_idx = int(np.argmax(probs))
                confidence = float(probs[winner_idx])
                winning_class = CLASSES[winner_idx].upper()

                if confidence >= threshold:
                    verdict = winning_class
                    score_display = confidence
                else:
                    verdict = "UNCERTAIN"
                    score_display = confidence

                # Update State and Batch Item Counts
                counts[verdict] += 1
                last_result = verdict
                last_score = score_display
                last_fresh_p = fresh_p
                last_rotten_p = rotten_p
                last_inference_ms = inference_ms
                last_scan_time = time.time()

                # Trigger flash banner
                flash_msg = f"{verdict} ({score_display * 100:.1f}%)"
                flash_time = time.time()

                # Update GPIO LEDs if available
                if led_controller:
                    led_controller.set_result(verdict)

                # Append to CSV log
                log_scan_event(
                    batch_id=batch_id,
                    result=verdict,
                    fresh_score=fresh_p,
                    rotten_score=rotten_p,
                    threshold=threshold,
                    inference_ms=inference_ms,
                )

                print(
                    f"[+] LOGGED ITEM #{sum(counts.values())}: {verdict:<9} | Score: {score_display * 100:.1f}% "
                    f"| Fresh: {fresh_p:.2f}, Rotten: {rotten_p:.2f} | Time: {inference_ms:.1f}ms | CSV Updated"
                )

    finally:
        if led_controller:
            led_controller.cleanup()
        camera.release()
        cv2.destroyAllWindows()
        print("[+] Application closed cleanly.")


# -----------------------------------------------------------------------------
# CLI Dispatcher
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fruit Freshness Detector - Dataset Collection & Live Inspection System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Operational mode to run")

    # Collect subcommand
    collect_parser = subparsers.add_parser("collect", help="Collect training/val/test crops for fruit specimens")
    collect_parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        required=True,
        help="Dataset split destination",
    )
    collect_parser.add_argument(
        "--label",
        choices=["fresh", "rotten"],
        required=True,
        help="Visual freshness condition label",
    )
    collect_parser.add_argument(
        "--id",
        required=True,
        help="Unique physical fruit specimen identity (e.g. apple001)",
    )
    collect_parser.add_argument(
        "--source",
        choices=["webcam", "pi"],
        default="webcam",
        help="Camera input source (default: webcam)",
    )
    collect_parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    collect_parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Dataset root directory (default: data/)",
    )

    # Scan subcommand
    scan_parser = subparsers.add_parser("scan", help="Run live real-time fruit inspection and classification")
    scan_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Confidence threshold for CERTAIN classification (default: {DEFAULT_THRESHOLD})",
    )
    scan_parser.add_argument(
        "--alert-count",
        type=int,
        default=DEFAULT_ALERT_COUNT,
        help=f"Number of rotten detections in batch before warning (default: {DEFAULT_ALERT_COUNT})",
    )
    scan_parser.add_argument(
        "--fruit",
        type=str,
        default=DEFAULT_SUPPORTED_FRUIT,
        help=f"Configured supported fruit variety (default: {DEFAULT_SUPPORTED_FRUIT})",
    )
    scan_parser.add_argument(
        "--source",
        choices=["webcam", "pi"],
        default="webcam",
        help="Camera input source (default: webcam)",
    )
    scan_parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    scan_parser.add_argument(
        "--auto-interval",
        type=float,
        default=0.5,
        help="Interval in seconds for automatic live camera classification (default: 0.5s; set 0 for continuous)",
    )
    scan_parser.add_argument(
        "--gpio",
        action="store_true",
        help="Enable GPIO status indicator LEDs (Raspberry Pi only)",
    )

    args = parser.parse_args()

    if args.command == "collect":
        run_collection(
            split=args.split,
            label=args.label,
            fruit_id=args.id,
            source=args.source,
            index=args.index,
            data_dir=args.data_dir,
        )
    elif args.command == "scan":
        run_scanner(
            threshold=args.threshold,
            alert_count=args.alert_count,
            fruit_name=args.fruit,
            auto_interval=args.auto_interval,
            source=args.source,
            index=args.index,
            enable_gpio=args.gpio,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
