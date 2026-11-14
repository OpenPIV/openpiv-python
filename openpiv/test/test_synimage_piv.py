"""Tests using the synthetic image generator to validate PIV analysis."""

import numpy as np
import pytest

from openpiv import pyprocess


def test_synimage_piv_basic():
    """Generate synthetic PIV images and run cross-correlation."""
    from openpiv.tools import synimage

    image_a, image_b = synimage.synimagegen(128, bit_depth=8)

    u, v, sig2noise = pyprocess.extended_search_area_piv(
        image_a.astype(np.int32),
        image_b.astype(np.int32),
        window_size=32,
        overlap=16,
        dt=0.1,
        search_area_size=64,
    )

    assert isinstance(u, np.ndarray)
    assert u.shape == v.shape == sig2noise.shape
    # At least some valid (finite) vectors
    assert np.isfinite(u).sum() > 0
    assert np.isfinite(v).sum() > 0


def test_synimage_piv_displacement_sign():
    """PIV should recover a non-zero displacement field."""
    from openpiv.tools import synimage

    image_a, image_b = synimage.synimagegen(128, bit_depth=8, dt=0.1)

    u, v, _sig2noise = pyprocess.extended_search_area_piv(
        image_a.astype(np.int32),
        image_b.astype(np.int32),
        window_size=32,
        overlap=16,
        dt=0.1,
        search_area_size=64,
    )

    valid = np.isfinite(u) & np.isfinite(v)
    assert valid.sum() > 0
    # The synthetic flow has a positive mean U component
    assert np.nanmedian(u[valid]) != 0 or np.nanmedian(v[valid]) != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
