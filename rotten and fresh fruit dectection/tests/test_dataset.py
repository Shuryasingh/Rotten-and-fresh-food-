"""Unit tests for dataset naming parsing, physical specimen isolation, and SHA-256 duplicate detection."""

from pathlib import Path
import tempfile
import pytest

from app import verify_specimen_isolation
from audit_dataset import compute_file_sha256, parse_fruit_identity


def test_parse_fruit_identity_valid():
    """Verify parsing of valid fruit filenames."""
    ok, fruit_id = parse_fruit_identity("apple001__view_top.jpg")
    assert ok is True
    assert fruit_id == "apple001"

    ok, fruit_id = parse_fruit_identity("specimen-42__shot99.png")
    assert ok is True
    assert fruit_id == "specimen-42"

    ok, fruit_id = parse_fruit_identity("BANANA-123__01.jpeg")
    assert ok is True
    assert fruit_id == "BANANA-123"


def test_parse_fruit_identity_invalid():
    """Verify rejection of filenames violating naming constraints."""
    # Missing '__' separator
    ok, err = parse_fruit_identity("apple001.jpg")
    assert ok is False
    assert "lacks the required '__'" in err

    # Space in fruit ID
    ok, err = parse_fruit_identity("apple 001__view.jpg")
    assert ok is False
    assert "invalid characters" in err

    # Empty fruit ID
    ok, err = parse_fruit_identity("__view.jpg")
    assert ok is False
    assert "empty" in err


def test_specimen_isolation_detection():
    """Verify that physical fruit identities crossing split boundaries are correctly flagged."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_root = Path(tmp_dir)

        # Create train and val directories
        train_fresh = data_root / "train" / "fresh"
        val_fresh = data_root / "val" / "fresh"
        train_fresh.mkdir(parents=True)
        val_fresh.mkdir(parents=True)

        # Place an image for 'apple001' in train/fresh
        (train_fresh / "apple001__shot01.jpg").write_text("dummy")

        # Check 1: Trying to add 'apple001' to val split should flag conflict with 'train'
        conflict = verify_specimen_isolation(target_split="val", fruit_id="apple001", data_dir=data_root)
        assert conflict == "train"

        # Check 2: Adding another shot of 'apple001' to train split should be allowed (same split)
        conflict_same = verify_specimen_isolation(target_split="train", fruit_id="apple001", data_dir=data_root)
        assert conflict_same is None

        # Check 3: A completely new specimen 'apple002' should have no conflict
        conflict_new = verify_specimen_isolation(target_split="val", fruit_id="apple002", data_dir=data_root)
        assert conflict_new is None


def test_sha256_duplicate_detection():
    """Verify file SHA-256 hash calculation and duplicate detection."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        file1 = Path(tmp_dir) / "sample1.jpg"
        file2 = Path(tmp_dir) / "sample2.jpg"
        file3 = Path(tmp_dir) / "sample3.jpg"

        file1.write_bytes(b"exact binary content for test")
        file2.write_bytes(b"exact binary content for test")
        file3.write_bytes(b"different content")

        hash1 = compute_file_sha256(file1)
        hash2 = compute_file_sha256(file2)
        hash3 = compute_file_sha256(file3)

        assert hash1 == hash2
        assert hash1 != hash3
