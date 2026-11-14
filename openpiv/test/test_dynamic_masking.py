"""Tests for dynamic masking using synthetic images."""

import numpy as np
import pytest

from openpiv import preprocess
from openpiv.tools import synimage


def test_dynamic_masking_edges_shape():
    image_a, _ = synimage.synimagegen(128, bit_depth=8)
    masked, mask = preprocess.dynamic_masking(image_a, method="edges")

    assert masked.shape == image_a.shape
    assert mask.shape == image_a.shape
    assert mask.dtype == np.uint8 or mask.dtype == bool or mask.dtype == np.bool_


def test_dynamic_masking_intensity_shape():
    image_a, _ = synimage.synimagegen(128, bit_depth=8)
    masked, mask = preprocess.dynamic_masking(image_a, method="intensity")

    assert masked.shape == image_a.shape
    assert mask.shape == image_a.shape


def test_dynamic_masking_masked_pixels_zero():
    image_a, _ = synimage.synimagegen(128, bit_depth=8)
    masked, mask = preprocess.dynamic_masking(image_a, method="edges")

    bool_mask = mask.astype(bool)
    if bool_mask.any():
        assert np.all(masked[bool_mask] == 0)


def test_dynamic_masking_invalid_method():
    image_a, _ = synimage.synimagegen(64, bit_depth=8)
    with pytest.raises(ValueError):
        preprocess.dynamic_masking(image_a, method="unknown")


def test_mask_coordinates_simple():
    mask = np.zeros((64, 64), dtype=int)
    mask[10:40, 10:40] = 1
    coords = preprocess.mask_coordinates(mask, tolerance=1.5, min_length=5)
    # Should return some coordinates (list or array)
    assert coords is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
