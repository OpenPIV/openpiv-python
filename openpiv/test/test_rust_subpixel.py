import numpy as np
import pytest

openpiv_rust = pytest.importorskip("openpiv_rust")
from openpiv import pyprocess


def test_rust_find_subpixel_peak_position_methods():
    """Verify all 3 subpixel methods match Python exactly."""
    corr = np.zeros((5, 5), dtype=np.float64)
    corr[2, 2] = 1.0
    corr[1, 2] = 0.9
    corr[3, 2] = 0.5
    corr[2, 1] = 0.6
    corr[2, 3] = 0.8

    for method in ["gaussian", "centroid", "parabolic"]:
        py_res = pyprocess.find_subpixel_peak_position(corr, subpixel_method=method)
        rust_res = openpiv_rust.find_subpixel_peak_position(corr, subpixel_method=method)
        assert np.allclose(py_res, rust_res, atol=1e-6)


def test_rust_find_subpixel_peak_position_boundary():
    """Verify boundary peaks return NaNs safely."""
    corr = np.zeros((5, 5), dtype=np.float64)
    corr[0, 0] = 1.0
    rust_res = openpiv_rust.find_subpixel_peak_position(corr)
    assert np.isnan(rust_res[0]) and np.isnan(rust_res[1])


def test_rust_find_subpixel_peak_position_invalid():
    """Verify invalid subpixel methods raise ValueError."""
    corr = np.ones((5, 5), dtype=np.float64)
    with pytest.raises(ValueError):
        openpiv_rust.find_subpixel_peak_position(corr, subpixel_method="invalid")


def test_rust_batch_correlation_to_displacement():
    """Verify batched correlation_to_displacement matches Python and has correct shape."""
    rng = np.random.default_rng(42)
    n_rows, n_cols = 10, 10
    n_wins = n_rows * n_cols
    corr = rng.random((n_wins, 16, 16), dtype=np.float64)

    for k in range(n_wins):
        pi = rng.integers(2, 14)
        pj = rng.integers(2, 14)
        corr[k, pi, pj] = 10.0

    u_rust, v_rust = openpiv_rust.batch_correlation_to_displacement(corr, n_rows, n_cols, "gaussian")
    assert u_rust.shape == (n_rows, n_cols)
    assert v_rust.shape == (n_rows, n_cols)

    peaks_i, peaks_j = openpiv_rust.batch_find_subpixel_peak_position(corr, "gaussian")
    assert len(peaks_i) == n_wins
    assert len(peaks_j) == n_wins
    assert np.allclose(peaks_j - 8.0, u_rust.ravel())
    assert np.allclose(peaks_i - 8.0, v_rust.ravel())
