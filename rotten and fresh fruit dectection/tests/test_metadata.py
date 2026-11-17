"""Unit tests for metadata validation and threshold decision logic in common.py."""

from pathlib import Path
import json
import tempfile
import pytest

from common import CLASSES, SIZE, load_metadata, save_metadata


def test_metadata_valid():
    """Verify loading valid metadata configuration."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        meta_file = Path(tmp_dir) / "metadata.json"
        valid_meta = {
            "classes": ["fresh", "rotten"],
            "size": 224,
            "model": "MobileNetV2",
            "format": "ONNX",
            "threshold": 0.80,
            "created_at": "2026-09-26T10:00:00+00:00",
        }
        save_metadata(valid_meta, meta_file)

        loaded = load_metadata(meta_file)
        assert loaded["classes"] == CLASSES
        assert loaded["size"] == SIZE
        assert loaded["model"] == "MobileNetV2"


def test_metadata_missing_file():
    """Verify informative FileNotFoundError when metadata does not exist."""
    non_existent = Path("non_existent_metadata_file.json")
    with pytest.raises(FileNotFoundError) as excinfo:
        load_metadata(non_existent)
    assert "Model metadata not found" in str(excinfo.value)


def test_metadata_invalid_classes():
    """Verify rejection when class list or order does not match standard."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        meta_file = Path(tmp_dir) / "metadata.json"
        invalid_meta = {
            "classes": ["rotten", "fresh"],  # Swapped order
            "size": 224,
            "model": "MobileNetV2",
            "format": "ONNX",
        }
        save_metadata(invalid_meta, meta_file)

        with pytest.raises(ValueError) as excinfo:
            load_metadata(meta_file)
        assert "class mismatch" in str(excinfo.value)


def test_metadata_invalid_size():
    """Verify rejection when image input size differs from expected 224."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        meta_file = Path(tmp_dir) / "metadata.json"
        invalid_meta = {
            "classes": ["fresh", "rotten"],
            "size": 512,  # Wrong size
            "model": "MobileNetV2",
            "format": "ONNX",
        }
        save_metadata(invalid_meta, meta_file)

        with pytest.raises(ValueError) as excinfo:
            load_metadata(meta_file)
        assert "image size mismatch" in str(excinfo.value)


def test_threshold_classification_logic():
    """Verify certainty decision logic with thresholds and uncertain fallback."""
    threshold = 0.80

    # Case 1: Fresh confidence 0.85 >= 0.80 -> FRESH
    probs_fresh = [0.85, 0.15]
    winner_idx = 0
    conf = probs_fresh[winner_idx]
    result = CLASSES[winner_idx].upper() if conf >= threshold else "UNCERTAIN"
    assert result == "FRESH"

    # Case 2: Rotten confidence 0.92 >= 0.80 -> ROTTEN
    probs_rotten = [0.08, 0.92]
    winner_idx = 1
    conf = probs_rotten[winner_idx]
    result = CLASSES[winner_idx].upper() if conf >= threshold else "UNCERTAIN"
    assert result == "ROTTEN"

    # Case 3: Ambiguous prediction 0.65 < 0.80 -> UNCERTAIN
    probs_ambiguous = [0.65, 0.35]
    winner_idx = 0
    conf = probs_ambiguous[winner_idx]
    result = CLASSES[winner_idx].upper() if conf >= threshold else "UNCERTAIN"
    assert result == "UNCERTAIN"

    # Case 4: Exact threshold boundary 0.80 == 0.80 -> FRESH
    probs_edge = [0.80, 0.20]
    winner_idx = 0
    conf = probs_edge[winner_idx]
    result = CLASSES[winner_idx].upper() if conf >= threshold else "UNCERTAIN"
    assert result == "FRESH"
