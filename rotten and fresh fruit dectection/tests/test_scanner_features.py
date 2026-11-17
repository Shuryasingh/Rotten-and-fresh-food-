"""Unit tests for trained ONNX predictor, automatic interval, and missing model guidance."""

import io
from pathlib import Path
import numpy as np
import pytest

from app import print_model_missing_guidance
from common import (
    CLASSES,
    MODEL_ONNX_PATH,
    Predictor,
)


def test_trained_predictor_loads_and_infers():
    """Verify that the exported fruit.onnx loads and produces calibrated probabilities."""
    assert MODEL_ONNX_PATH.exists(), "fruit.onnx artifact must exist"

    predictor = Predictor()
    dummy_bgr = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    probs = predictor.predict(dummy_bgr)

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (len(CLASSES),)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-4)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_missing_model_guidance_output(capsys):
    """Verify clear, actionable setup guidance when model is missing."""
    fake_error = FileNotFoundError("Missing artifacts/fruit.onnx")
    print_model_missing_guidance(fake_error)

    captured = capsys.readouterr().out
    assert "MODEL NOT FOUND" in captured
    assert "download_dataset.py" in captured
    assert "train.py" in captured
    assert "audit_dataset.py" in captured


def test_predictor_missing_file_raises_filenotfound():
    """Verify Predictor raises FileNotFoundError when given non-existent path."""
    with pytest.raises(FileNotFoundError) as exc_info:
        Predictor(onnx_path=Path("non_existent_model_file.onnx"))
    assert "No trained model found" in str(exc_info.value)
