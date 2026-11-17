"""Dataset integrity and hygiene audit tool.

Validates the dataset structure against project rules:
1. All 6 split folders exist (train/val/test x fresh/rotten)
2. Images are readable and valid
3. Fruit identity format (<id>__<name>.jpg) is enforced
4. Physical specimen isolation: No fruit identity appears in multiple splits
5. No exact duplicate image files cross split boundaries (SHA-256 validation)
6. Reports sample counts and physical specimen counts
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2

from common import CLASSES, DATA_DIR, FRUIT_ID_REGEX, logger

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "val", "test"]


def compute_file_sha256(filepath: Path) -> str:
    """Calculate the SHA-256 hash of a file for exact-duplicate detection."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_fruit_identity(filename: str) -> Tuple[bool, str]:
    """Extract physical fruit identity prefix before '__'.

    Example:
        'apple001__view01.jpg' -> (True, 'apple001')
        'apple001.jpg' -> (False, 'Missing double-underscore delimiter')
        'apple 001__01.jpg' -> (False, 'Invalid characters in fruit ID')
    """
    stem = Path(filename).stem
    if "__" not in stem:
        return False, f"Filename '{filename}' lacks the required '__' delimiter separating fruit ID from unique view."

    fruit_id = stem.split("__", 1)[0]
    if not fruit_id:
        return False, f"Fruit ID prefix before '__' is empty in '{filename}'."

    if not FRUIT_ID_REGEX.match(fruit_id):
        return False, f"Fruit ID '{fruit_id}' contains invalid characters. Use letters, numbers, and hyphens only."

    return True, fruit_id


def audit_dataset(data_dir: Path = DATA_DIR) -> bool:
    """Run comprehensive dataset audit and return True if passed, False otherwise."""
    print("=" * 70)
    print("FRUIT FRESHNESS DATASET INTEGRITY AUDIT")
    print("=" * 70)
    print(f"Checking dataset root: {data_dir.resolve()}\n")

    errors: List[str] = []
    warnings: List[str] = []

    # 1. Verify directory hierarchy
    for split in SPLITS:
        for cls_name in CLASSES:
            folder = data_dir / split / cls_name
            if not folder.exists():
                errors.append(f"Missing required dataset folder: {folder.relative_to(data_dir.parent)}")

    if errors:
        print("[-] DIRECTORY STRUCTURE ERRORS:")
        for err in errors:
            print(f"    - {err}")
        print("\nFix: Ensure all train/val/test folders exist for both fresh and rotten classes.")
        return False

    # Tracking metrics
    counts: Dict[str, Dict[str, int]] = {s: {c: 0 for c in CLASSES} for s in SPLITS}
    split_identities: Dict[str, Set[str]] = {s: set() for s in SPLITS}
    identity_to_splits: Dict[str, Set[str]] = defaultdict(set)
    identity_to_files: Dict[str, List[Path]] = defaultdict(list)
    hash_to_split: Dict[str, Tuple[str, Path]] = {}

    total_images = 0

    # 2. Inspect each split and class
    for split in SPLITS:
        for cls_name in CLASSES:
            folder = data_dir / split / cls_name
            files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
            counts[split][cls_name] = len(files)
            total_images += len(files)

            for img_path in files:
                # Check 2a: Image readability and dimensions
                img = cv2.imread(str(img_path))
                if img is None:
                    errors.append(f"Corrupt or unreadable image: {img_path.relative_to(data_dir.parent)}")
                    continue
                if img.shape[0] < 32 or img.shape[1] < 32:
                    warnings.append(
                        f"Very small image ({img.shape[1]}x{img.shape[0]}): {img_path.relative_to(data_dir.parent)}"
                    )

                # Check 2b: Identity format
                valid_id, fruit_id_or_err = parse_fruit_identity(img_path.name)
                if not valid_id:
                    errors.append(f"Naming format violation: {fruit_id_or_err}")
                else:
                    fruit_id = fruit_id_or_err
                    split_identities[split].add(fruit_id)
                    identity_to_splits[fruit_id].add(split)
                    identity_to_files[fruit_id].append(img_path)

                # Check 2c: Exact SHA-256 duplicate content crossing splits
                file_hash = compute_file_sha256(img_path)
                if file_hash in hash_to_split:
                    other_split, other_path = hash_to_split[file_hash]
                    if other_split != split:
                        errors.append(
                            f"Exact duplicate image content found across splits!\n"
                            f"      File 1: {img_path.relative_to(data_dir.parent)} ({split})\n"
                            f"      File 2: {other_path.relative_to(data_dir.parent)} ({other_split})"
                        )
                else:
                    hash_to_split[file_hash] = (split, img_path)

    # 3. Check for physical fruit identity leakage across splits
    for fruit_id, splits_found in identity_to_splits.items():
        if len(splits_found) > 1:
            file_list_str = ", ".join(str(p.relative_to(data_dir.parent)) for p in identity_to_files[fruit_id][:4])
            errors.append(
                f"Physical fruit identity leakage: ID '{fruit_id}' appears in multiple splits ({list(splits_found)}).\n"
                f"      Examples: {file_list_str}...\n"
                f"      Rule: All views of the same physical fruit specimen must belong to ONE split only."
            )

    # 4. Print Summary Table
    print("DATASET SUMMARY TABLE:")
    print("-" * 70)
    print(f"{'Split':<10} | {'Fresh Images':<14} | {'Rotten Images':<14} | {'Total':<8} | {'Unique Fruits':<14}")
    print("-" * 70)
    for split in SPLITS:
        fresh_n = counts[split]["fresh"]
        rotten_n = counts[split]["rotten"]
        total_s = fresh_n + rotten_n
        unique_f = len(split_identities[split])
        print(f"{split:<10} | {fresh_n:<14} | {rotten_n:<14} | {total_s:<8} | {unique_f:<14}")
    print("-" * 70)
    print(f"{'TOTAL':<10} | {sum(counts[s]['fresh'] for s in SPLITS):<14} | {sum(counts[s]['rotten'] for s in SPLITS):<14} | {total_images:<8} | {len(identity_to_splits):<14}")
    print("-" * 70)

    # 5. Handle empty dataset notice
    if total_images == 0:
        print("\n[!] NOTICE: Dataset is currently empty.")
        print("    No images have been placed in data/train, data/val, or data/test yet.")
        print("    This is expected on a fresh checkout.")
        print("\nNext steps to collect images:")
        print("  1. Test your webcam: python camera_test.py")
        print("  2. Collect fresh samples:  python app.py collect --split train --label fresh --id apple001")
        print("  3. Collect rotten samples: python app.py collect --split train --label rotten --id apple101")
        print("  4. Collect independent validation and test specimens.")
        print("  5. Re-run: python audit_dataset.py\n")
        return False

    # 6. Report Warnings
    if warnings:
        print("\n[!] WARNINGS (Non-critical):")
        for warn in warnings:
            print(f"    - {warn}")

    # 7. Report Errors and Final Verdict
    if errors:
        print("\n[-] AUDIT FAILED - Violations Detected:")
        for err in errors:
            print(f"    - {err}")
        print("\nFix all reported errors before proceeding with training.")
        return False

    # Check minimum split presence
    for split in SPLITS:
        for cls_name in CLASSES:
            if counts[split][cls_name] == 0:
                print(f"\n[!] Split '{split}/{cls_name}' has 0 images.")
                print(f"    For reliable training/evaluation, collect samples for all splits.")

    print("\n[+] AUDIT PASSED: Dataset structure, fruit identities, and isolation rules are valid.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit fruit freshness dataset for integrity, naming conventions, and specimen isolation."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Path to data directory (default: data/)",
    )
    args = parser.parse_args()

    success = audit_dataset(args.data_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
