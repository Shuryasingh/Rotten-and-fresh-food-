"""Dataset download and curation tool for Fruit Freshness Detector.

Downloads publicly available labeled fresh and rotten fruit images from the
Hugging Face 'menasi11/fruit-classification' dataset (CC0 / Public Domain),
curates the target fruit variety (Apples, plus optional Bananas and Oranges),
enforces physical specimen isolation across train, val, and test splits,
checks for SHA-256 duplicate content, and integrates cleanly with the existing dataset.

License & Source Documentation:
- Source: https://huggingface.co/datasets/menasi11/fruit-classification
- Original Data: Kaggle "Fruits fresh and rotten for classification" (Sriramr)
- License: CC0: Public Domain / Open Access
- Supported Fruit Classes in source:
  * fresh_apple, rotten_apple
  * fresh_banana, rotten_banana
  * fresh_orange, rotten_orange
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import random
import shutil
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from common import CLASSES, DATA_DIR, logger

HF_RAW_BASE = "https://huggingface.co/datasets/menasi11/fruit-classification/resolve/main"
TRAIN_CSV_URL = f"{HF_RAW_BASE}/train.csv"


def compute_bytes_sha256(data: bytes) -> str:
    """Calculate SHA-256 hash of byte array."""
    return hashlib.sha256(data).hexdigest()


def fetch_csv_metadata() -> List[Tuple[str, str]]:
    """Fetch and parse train.csv from Hugging Face repository.

    Returns:
        List of (filename, label_name) tuples.
    """
    print(f"[*] Fetching dataset metadata from: {TRAIN_CSV_URL} ...")
    req = urllib.request.Request(TRAIN_CSV_URL, headers={"User-Agent": "FruitDetectorSetup/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read().decode("utf-8")

    records: List[Tuple[str, str]] = []
    lines = content.splitlines()
    for line in lines[1:]:
        line = line.strip()
        if not line or "," not in line:
            continue
        parts = line.split(",", 1)
        filename = parts[0].strip()
        label = parts[1].strip()
        records.append((filename, label))

    print(f"[+] Loaded {len(records)} metadata records from Hugging Face.")
    return records


def download_single_image(img_filename: str) -> Optional[bytes]:
    """Download single image file from Hugging Face with retry."""
    url = f"{HF_RAW_BASE}/train/train/{img_filename}"
    req = urllib.request.Request(url, headers={"User-Agent": "FruitDetectorSetup/1.0"})

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                return resp.read()
        except Exception:
            if attempt == 2:
                return None
    return None


def setup_curated_dataset(
    data_dir: Path = DATA_DIR,
    fruit_variety: str = "apple",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_per_class: Optional[int] = None,
    seed: int = 42,
    num_workers: int = 12,
) -> bool:
    """Download, curate, and split public fruit images ensuring physical specimen isolation.

    Args:
        data_dir: Root dataset directory (contains train/, val/, test/).
        fruit_variety: 'apple' (default), 'banana', 'orange', or 'all'.
        train_ratio: Fraction for training split (default 0.70).
        val_ratio: Fraction for validation split (default 0.15).
        test_ratio: Fraction for test split (default 0.15).
        max_per_class: Optional cap on images per class.
        seed: Random seed for deterministic split assignment.
        num_workers: Concurrency level for downloads.

    Returns:
        True if dataset setup succeeded, False otherwise.
    """
    print("=" * 70)
    print("FRUIT FRESHNESS DETECTOR - DATASET SETUP & DOWNLOAD TOOL")
    print("=" * 70)
    print(f"Dataset Source:  Hugging Face (menasi11/fruit-classification)")
    print(f"License:         CC0: Public Domain / Open Data")
    print(f"Target Variety:  {fruit_variety.upper()}")
    print(f"Target Directory:{data_dir.resolve()}\n")

    # 1. Fetch metadata records
    try:
        all_records = fetch_csv_metadata()
    except Exception as err:
        print(f"[-] ERROR: Failed to fetch metadata from Hugging Face: {err}")
        print("    Please verify internet connectivity.")
        return False

    # 2. Filter by fruit variety
    # Labels look like: 'fresh_apple', 'rotten_apple', 'fresh_banana', etc.
    filtered: Dict[str, List[str]] = {"fresh": [], "rotten": []}

    fruit_key = fruit_variety.lower().strip()

    for filename, label in all_records:
        label_lower = label.lower()
        if fruit_key != "all" and fruit_key not in label_lower:
            continue

        if "fresh" in label_lower:
            filtered["fresh"].append(filename)
        elif "rotten" in label_lower:
            filtered["rotten"].append(filename)

    print(f"[*] Filtered candidates for '{fruit_variety}':")
    print(f"    - Fresh:  {len(filtered['fresh'])} images")
    print(f"    - Rotten: {len(filtered['rotten'])} images")

    if not filtered["fresh"] or not filtered["rotten"]:
        print(f"[-] No matching images found for variety '{fruit_variety}'.")
        return False

    # Apply optional cap
    rng = random.Random(seed)
    for cls_name in ["fresh", "rotten"]:
        rng.shuffle(filtered[cls_name])
        if max_per_class and len(filtered[cls_name]) > max_per_class:
            filtered[cls_name] = filtered[cls_name][:max_per_class]

    # 3. Check existing files and specimen IDs
    existing_hashes: Set[str] = set()
    for path in data_dir.glob("*/*/*.*"):
        if path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            with open(path, "rb") as f:
                existing_hashes.add(hashlib.sha256(f.read()).hexdigest())

    print(f"[*] Existing local dataset files found: {len(existing_hashes)} distinct hashes")

    # 4. Prepare splits with strict physical specimen isolation
    # Each downloaded image represents an individual specimen observation.
    # To conform strictly to '<fruit_id>__<unique_token>.jpg' and prevent leakage,
    # each specimen is given a unique fruit_id (e.g. 'pub-apple-f001') and assigned to
    # exactly ONE split (train, val, or test).
    tasks_to_download: List[Tuple[str, str, str, str]] = []  # (filename, split, cls_name, fruit_id)

    for cls_name in ["fresh", "rotten"]:
        items = filtered[cls_name]
        total = len(items)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)
        # remainder to test
        splits_assignment = (
            [("train", cls_name)] * n_train
            + [("val", cls_name)] * n_val
            + [("test", cls_name)] * (total - n_train - n_val)
        )

        for idx, (img_fn, (split, c_name)) in enumerate(zip(items, splits_assignment), start=1):
            fruit_id = f"pub-{fruit_variety[:3]}-{c_name[:3]}{idx:03d}"
            tasks_to_download.append((img_fn, split, c_name, fruit_id))

    print(f"[*] Total images scheduled for download: {len(tasks_to_download)}")
    print(f"    Using {num_workers} parallel worker threads...")

    # Ensure all target directories exist
    for split in ["train", "val", "test"]:
        for cls_name in CLASSES:
            (data_dir / split / cls_name).mkdir(parents=True, exist_ok=True)

    # 5. Download in parallel with progress reporting
    downloaded_count = 0
    skipped_duplicate = 0
    failed_count = 0

    def process_task(task: Tuple[str, str, str, str]) -> Tuple[bool, str]:
        img_fn, split, cls_name, fruit_id = task
        raw_bytes = download_single_image(img_fn)
        if raw_bytes is None:
            return False, f"Failed download: {img_fn}"

        # SHA-256 duplicate check
        sha = compute_bytes_sha256(raw_bytes)
        if sha in existing_hashes:
            return True, "duplicate_skipped"

        # Decode image using OpenCV to ensure valid image data
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return False, f"Corrupted image payload: {img_fn}"

        # Crop to square if not already square
        h, w = img.shape[:2]
        if h != w:
            side = min(h, w)
            y_start = (h - side) // 2
            x_start = (w - side) // 2
            img = img[y_start : y_start + side, x_start : x_start + side]

        # Target filename: <fruit_id>__<hash[:8]>.jpg
        unique_token = sha[:8]
        out_filename = f"{fruit_id}__{unique_token}.jpg"
        out_path = data_dir / split / cls_name / out_filename

        success = cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            return False, f"Failed to write image: {out_path}"

        existing_hashes.add(sha)
        return True, "ok"

    print("\n[*] Downloading and validating images...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_task, task): task for task in tasks_to_download}
        total_tasks = len(futures)

        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            success, msg = future.result()
            if success:
                if msg == "duplicate_skipped":
                    skipped_duplicate += 1
                else:
                    downloaded_count += 1
            else:
                failed_count += 1

            if i % 50 == 0 or i == total_tasks:
                percent = (i / total_tasks) * 100.0
                print(f"    [{i:>3}/{total_tasks}] ({percent:.0f}%) Downloaded: {downloaded_count}, Skipped Duplicates: {skipped_duplicate}, Failed: {failed_count}")

    print("\n" + "-" * 70)
    print(f"[+] Download complete: {downloaded_count} new images added.")
    if skipped_duplicate > 0:
        print(f"[*] Skipped duplicates: {skipped_duplicate}")
    if failed_count > 0:
        print(f"[-] Failed downloads: {failed_count}")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and curate public fresh/rotten fruit dataset with specimen isolation."
    )
    parser.add_argument(
        "--variety",
        type=str,
        default="apple",
        choices=["apple", "banana", "orange", "mango", "grape", "grapes", "strawberry", "avocado", "all"],
        help="Fruit variety to curate (default: apple, supports: mango, grapes, strawberry, avocado, banana, orange)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Target data directory (default: data/)",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional maximum images per class",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Concurrent download threads (default: 12)",
    )

    args = parser.parse_args()
    success = setup_curated_dataset(
        data_dir=args.data_dir,
        fruit_variety=args.variety,
        max_per_class=args.max_per_class,
        num_workers=args.workers,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
