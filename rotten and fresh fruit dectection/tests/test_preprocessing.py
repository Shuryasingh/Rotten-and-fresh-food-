"""Unit tests for image tensor preprocessing and color space conversion in common.py."""

import numpy as np
import pytest

from common import IMAGENET_MEAN, IMAGENET_STD, SIZE, tensor_image


def test_tensor_image_shape_and_type():
    """Verify output shape, dtype, and memory layout of preprocessed tensor."""
    # Arbitrary BGR crop of size 384x384
    bgr = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)
    tensor = tensor_image(bgr)

    assert tensor.shape == (1, 3, SIZE, SIZE)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous


def test_tensor_image_bgr_to_rgb_conversion():
    """Verify BGR -> RGB color channel conversion during tensor generation."""
    # Create an image that is purely RED in RGB -> in BGR this is [0, 0, 255] (B=0, G=0, R=255)
    pure_red_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    pure_red_bgr[:, :, 2] = 255  # Red channel in BGR

    tensor = tensor_image(pure_red_bgr)
    # After BGR->RGB conversion and scaling to [0,1]:
    # R channel should be 1.0 -> normalized: (1.0 - mean[0]) / std[0]
    # G channel should be 0.0 -> normalized: (0.0 - mean[1]) / std[1]
    # B channel should be 0.0 -> normalized: (0.0 - mean[2]) / std[2]

    expected_r = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    expected_g = (0.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    expected_b = (0.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]

    # Check center pixel
    assert np.isclose(tensor[0, 0, 112, 112], expected_r, atol=1e-4)
    assert np.isclose(tensor[0, 1, 112, 112], expected_g, atol=1e-4)
    assert np.isclose(tensor[0, 2, 112, 112], expected_b, atol=1e-4)


def test_tensor_image_invalid_input():
    """Verify exception handling for empty or None image inputs."""
    with pytest.raises(ValueError):
        tensor_image(None)

    with pytest.raises(ValueError):
        tensor_image(np.array([]))
