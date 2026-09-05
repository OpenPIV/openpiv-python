import pytest
import numpy as np
import scipy.signal
import scipy.fft
from pathlib import Path
from importlib.resources import files
from openpiv import pyprocess, tools, validation, filters
openpiv_rust = pytest.importorskip("openpiv_rust")

@pytest.mark.parametrize("win_size", [16, 32, 64])
@pytest.mark.parametrize("norm", [True, False])
def test_circular_correlation_numerical_accuracy(win_size, norm):
    np.random.seed(42)
    n_wins = 20
    a = np.random.rand(n_wins, win_size, win_size)
    b = np.random.rand(n_wins, win_size, win_size)

    # Scipy reference
    scipy_corr = pyprocess.fft_correlate_images(
        a, b, correlation_method="circular", normalized_correlation=norm
    )

    if norm:
        a_norm = pyprocess.normalize_intensity(a)
        b_norm = pyprocess.normalize_intensity(b)
        rust_corr = openpiv_rust.fft_correlate_circular(a_norm, b_norm, normalized_correlation=True)
    else:
        rust_corr = openpiv_rust.fft_correlate_circular(a, b, normalized_correlation=False)

    max_diff = np.max(np.abs(scipy_corr - rust_corr))
    assert max_diff < 1e-10, f"Max difference {max_diff} exceeded tolerance for win_size={win_size}"

def test_rectangular_windows():
    np.random.seed(123)
    a = np.random.rand(10, 32, 64)
    b = np.random.rand(10, 32, 64)

    a_norm = pyprocess.normalize_intensity(a)
    b_norm = pyprocess.normalize_intensity(b)

    scipy_corr = pyprocess.fft_correlate_images(a, b, correlation_method="circular", normalized_correlation=True)
    rust_corr = openpiv_rust.fft_correlate_circular(a_norm, b_norm, normalized_correlation=True)

    max_diff = np.max(np.abs(scipy_corr - rust_corr))
    assert max_diff < 1e-10

def test_fast_batch_cross_correlation_mode_full():
    np.random.seed(456)
    n_wins, win_h, win_w = 15, 24, 24
    a = np.random.rand(n_wins, win_h, win_w)
    b = np.random.rand(n_wins, win_h, win_w)

    scipy_res = np.array([
        scipy.signal.correlate(a[i], b[i], mode="full") for i in range(n_wins)
    ])
    rust_res = openpiv_rust.fast_batch_cross_correlation(a, b)

    max_diff = np.max(np.abs(scipy_res - rust_res))
    assert max_diff < 1e-10

def test_strided_non_contiguous_inputs():
    """Verify Rust mode handles non-contiguous arrays without panicking."""
    a = np.random.rand(10, 32, 64)[:, :, ::2] # non-contiguous slice
    b = np.random.rand(10, 32, 64)[:, :, ::2]
    assert not a.flags.c_contiguous, "Array should be non-contiguous"

    rust_res = openpiv_rust.fft_correlate_circular(a, b, normalized_correlation=False)
    assert rust_res.shape == (10, 32, 32)

def test_known_displacements():
    """Verify integer peak detection recovers known displacement."""
    n_wins = 5
    win_size = 32
    shift_y, shift_x = 3, -2

    a = np.zeros((n_wins, win_size, win_size))
    b = np.zeros((n_wins, win_size, win_size))
    for i in range(n_wins):
        a[i, 14, 14] = 10.0
        b[i, 14 + shift_y, 14 + shift_x] = 10.0

    corr = openpiv_rust.fft_correlate_circular(a, b, normalized_correlation=False)
    center = win_size // 2

    for i in range(n_wins):
        (peak_y, peak_x), peak_val = pyprocess.find_first_peak(corr[i])
        dx = peak_x - center
        dy = peak_y - center
        assert dx == shift_x, f"Expected dx={shift_x}, got {dx}"
        assert dy == shift_y, f"Expected dy={shift_y}, got {dy}"

def test_real_piv_data_velocity_match():
    """Run full PIV pipeline on test1 dataset comparing Scipy vs Rust correlation."""
    path = files('openpiv') / "data" / "test1"
    frame_a = tools.imread(path / "exp1_001_a.bmp").astype(np.int32)
    frame_b = tools.imread(path / "exp1_001_b.bmp").astype(np.int32)

    window_size = 32
    overlap = 16
    search_area_size = 32

    # Scipy PIV
    u_scipy, v_scipy, s2n_scipy = pyprocess.extended_search_area_piv(
        frame_a, frame_b,
        window_size=window_size,
        overlap=overlap,
        search_area_size=search_area_size,
        correlation_method="circular",
        sig2noise_method="peak2peak",
    )

    # Rust PIV: extract windows and correlate via Rust
    aa = pyprocess.sliding_window_array(frame_a, (search_area_size, search_area_size), (overlap, overlap))
    bb = pyprocess.sliding_window_array(frame_b, (search_area_size, search_area_size), (overlap, overlap))
    rust_corr = openpiv_rust.fft_correlate_circular(aa.astype(float), bb.astype(float), normalized_correlation=False)

    n_rows, n_cols = pyprocess.get_field_shape(frame_a.shape, (search_area_size, search_area_size), (overlap, overlap))
    u_rust, v_rust = pyprocess.correlation_to_displacement(rust_corr, n_rows, n_cols)

    diff_u = np.nanmax(np.abs(u_scipy - u_rust))
    diff_v = np.nanmax(np.abs(v_scipy - v_rust))
    assert diff_u < 1e-8, f"Velocity difference in U: {diff_u}"
    assert diff_v < 1e-8, f"Velocity difference in V: {diff_v}"

def test_safe_handling_shape_mismatch():
    """Verify that shape mismatches raise ValueError safely rather than panicking."""
    a = np.random.rand(5, 32, 32)
    b = np.random.rand(5, 32, 16)
    with pytest.raises(ValueError, match="Shape mismatch"):
        openpiv_rust.fft_correlate_circular(a, b)

    with pytest.raises(ValueError, match="Shape mismatch"):
        openpiv_rust.fast_batch_cross_correlation(a, b)

def test_safe_handling_empty_arrays():
    """Verify that empty inputs raise ValueError safely."""
    a = np.zeros((0, 32, 32))
    b = np.zeros((0, 32, 32))
    with pytest.raises(ValueError, match="non-zero"):
        openpiv_rust.fft_correlate_circular(a, b)

def test_safe_handling_fortran_and_transposed():
    """Verify Fortran-ordered and transposed arrays work correctly."""
    rng = np.random.default_rng(123)
    a_c = rng.standard_normal((4, 32, 32))
    b_c = rng.standard_normal((4, 32, 32))

    # Convert to Fortran contiguous
    a_f = np.asfortranarray(a_c)
    b_f = np.asfortranarray(b_c)

    res_c = openpiv_rust.fft_correlate_circular(a_c, b_c)
    res_f = openpiv_rust.fft_correlate_circular(a_f, b_f)
    assert np.allclose(res_c, res_f, atol=1e-12)

def test_fft_correlate_linear_alias():
    """Verify fft_correlate_linear alias produces identical results to fast_batch_cross_correlation."""
    a = np.random.rand(3, 16, 16)
    b = np.random.rand(3, 16, 16)
    res1 = openpiv_rust.fast_batch_cross_correlation(a, b)
    res2 = openpiv_rust.fft_correlate_linear(a, b)
    assert np.array_equal(res1, res2)


def test_fft_correlate_images_linear_rust():
    """Verify pyprocess.fft_correlate_images works with backend='rust' and linear correlation."""
    a = np.random.rand(4, 32, 32)
    b = np.random.rand(4, 32, 32)
    corr_scipy = pyprocess.fft_correlate_images(a, b, correlation_method="linear", backend="scipy", normalized_correlation=False)
    corr_rust = pyprocess.fft_correlate_images(a, b, correlation_method="linear", backend="rust", normalized_correlation=False)
    assert corr_rust.shape == corr_scipy.shape
    assert np.allclose(corr_scipy, corr_rust, atol=1e-10)


