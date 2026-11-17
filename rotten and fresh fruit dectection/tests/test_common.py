"""Unit tests for square crop geometry and framing logic in common.py."""

import numpy as np
import pytest

from common import square


def test_square_crop_landscape_frame():
    """Verify centered square crop on standard 640x480 landscape frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cropped, (x, y, side) = square(frame, coverage=0.80)

    # 80% of 480 is 384
    expected_side = int(480 * 0.80)
    expected_x = (640 - expected_side) // 2
    expected_y = (480 - expected_side) // 2

    assert side == expected_side
    assert x == expected_x
    assert y == expected_y
    assert cropped.shape == (expected_side, expected_side, 3)


def test_square_crop_square_frame():
    """Verify centered square crop on 500x500 square frame."""
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    cropped, (x, y, side) = square(frame, coverage=0.80)

    expected_side = 400
    expected_x = 50
    expected_y = 50

    assert side == expected_side
    assert x == expected_x
    assert y == expected_y
    assert cropped.shape == (400, 400, 3)


def test_square_crop_portrait_frame():
    """Verify centered square crop on 480x640 portrait frame."""
    frame = np.zeros((640, 480, 3), dtype=np.uint8)
    cropped, (x, y, side) = square(frame, coverage=0.80)

    expected_side = int(480 * 0.80)
    expected_x = (480 - expected_side) // 2
    expected_y = (640 - expected_side) // 2

    assert side == expected_side
    assert x == expected_x
    assert y == expected_y
    assert cropped.shape == (expected_side, expected_side, 3)


def test_square_crop_invalid_frame():
    """Verify exception handling for empty or None frame input."""
    with pytest.raises(ValueError):
        square(None)

    with pytest.raises(ValueError):
        square(np.array([]))
