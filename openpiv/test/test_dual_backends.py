import pytest
import numpy as np
from pathlib import Path
from importlib.resources import files
from openpiv import pyprocess, piv, tools, windef

openpiv_rust = pytest.importorskip("openpiv_rust")


@pytest.fixture
def image_pair():
    """Load sample test images for dual-backend testing."""
    im1_path = Path(files("openpiv.data").joinpath("test1/exp1_001_a.bmp"))
    im2_path = Path(files("openpiv.data").joinpath("test1/exp1_001_b.bmp"))
    im1 = tools.imread(im1_path)
    im2 = tools.imread(im2_path)
    return im1, im2


def test_extended_search_area_piv_dual_backends(image_pair):
    """Verify numerical parity of extended_search_area_piv across auto, rust, and scipy backends."""
    im1, im2 = image_pair
    # Crop to small region for fast testing
    a = im1[:128, :128].astype(np.int32)
    b = im2[:128, :128].astype(np.int32)

    u_scipy, v_scipy, s2n_scipy = pyprocess.extended_search_area_piv(
        a, b, window_size=32, overlap=16, backend="scipy"
    )
    u_rust, v_rust, s2n_rust = pyprocess.extended_search_area_piv(
        a, b, window_size=32, overlap=16, backend="rust"
    )
    u_auto, v_auto, s2n_auto = pyprocess.extended_search_area_piv(
        a, b, window_size=32, overlap=16, backend="auto"
    )

    # Rust and Auto should produce identical results
    assert np.allclose(u_rust, u_auto, equal_nan=True)
    assert np.allclose(v_rust, v_auto, equal_nan=True)
    assert np.allclose(s2n_rust, s2n_auto, equal_nan=True)

    # Dual backend parity with SciPy reference
    assert np.allclose(u_scipy, u_rust, equal_nan=True, atol=1e-8)
    assert np.allclose(v_scipy, v_rust, equal_nan=True, atol=1e-8)
    assert np.allclose(s2n_scipy, s2n_rust, equal_nan=True, atol=1e-8)


@pytest.mark.parametrize("sig2noise_method", ["peak2peak", "peak2mean", None])
def test_extended_search_area_piv_sig2noise_options(image_pair, sig2noise_method):
    """Verify sig2noise options match between Rust and SciPy."""
    im1, im2 = image_pair
    a = im1[:128, :128].astype(np.int32)
    b = im2[:128, :128].astype(np.int32)

    u_scipy, v_scipy, s2n_scipy = pyprocess.extended_search_area_piv(
        a, b, window_size=32, overlap=16, sig2noise_method=sig2noise_method, backend="scipy"
    )
    u_rust, v_rust, s2n_rust = pyprocess.extended_search_area_piv(
        a, b, window_size=32, overlap=16, sig2noise_method=sig2noise_method, backend="rust"
    )

    assert np.allclose(u_scipy, u_rust, equal_nan=True, atol=1e-8)
    assert np.allclose(v_scipy, v_rust, equal_nan=True, atol=1e-8)
    if sig2noise_method is not None:
        assert np.allclose(s2n_scipy, s2n_rust, equal_nan=True, atol=1e-8)
    else:
        assert np.all(np.isnan(s2n_scipy))
        assert np.all(np.isnan(s2n_rust))


def test_simple_piv_dual_backends(image_pair):
    """Verify high-level simple_piv yields identical results for auto, rust, and scipy."""
    im1, im2 = image_pair
    a = im1[:128, :128]
    b = im2[:128, :128]

    x_s, y_s, u_s, v_s, s2n_s = piv.simple_piv(a, b, window_size=32, overlap=16, plot=False, backend="scipy")
    x_r, y_r, u_r, v_r, s2n_r = piv.simple_piv(a, b, window_size=32, overlap=16, plot=False, backend="rust")
    x_a, y_a, u_a, v_a, s2n_a = piv.simple_piv(a, b, window_size=32, overlap=16, plot=False, backend="auto")

    assert np.array_equal(x_s, x_r)
    assert np.array_equal(y_s, y_r)
    assert np.allclose(u_s, u_r, equal_nan=True, atol=1e-8)
    assert np.allclose(v_s, v_r, equal_nan=True, atol=1e-8)
    assert np.allclose(s2n_s, s2n_r, equal_nan=True, atol=1e-8)

    assert np.allclose(u_r, u_a, equal_nan=True)
    assert np.allclose(v_r, v_a, equal_nan=True)
    assert np.allclose(s2n_r, s2n_a, equal_nan=True)


def test_process_pair_dual_backends(image_pair):
    """Verify process_pair pipeline with validation works identically across backends."""
    im1, im2 = image_pair
    a = im1[:128, :128]
    b = im2[:128, :128]

    x_s, y_s, u_s, v_s, mask_s = piv.process_pair(a, b, window_size=32, overlap=16, plot=False, backend="scipy")
    x_r, y_r, u_r, v_r, mask_r = piv.process_pair(a, b, window_size=32, overlap=16, plot=False, backend="rust")
    x_a, y_a, u_a, v_a, mask_a = piv.process_pair(a, b, window_size=32, overlap=16, plot=False, backend="auto")

    assert np.array_equal(mask_s, mask_r)
    assert np.array_equal(mask_r, mask_a)
    assert np.allclose(u_s, u_r, equal_nan=True, atol=1e-8)
    assert np.allclose(v_s, v_r, equal_nan=True, atol=1e-8)


def test_windef_simple_multipass_dual_backends(image_pair):
    """Verify windef.simple_multipass produces identical results between scipy and rust."""
    im1, im2 = image_pair
    a = im1[:128, :128]
    b = im2[:128, :128]

    settings = windef.PIVSettings()
    settings.windowsizes = (32, 16)
    settings.overlap = (16, 8)
    settings.num_iterations = 1
    settings.show_all_plots = False
    settings.show_plot = False

    settings.backend = "scipy"
    x_s, y_s, u_s, v_s, mask_s = windef.simple_multipass(a, b, settings)

    settings.backend = "rust"
    x_r, y_r, u_r, v_r, mask_r = windef.simple_multipass(a, b, settings)

    settings.backend = "auto"
    x_a, y_a, u_a, v_a, mask_a = windef.simple_multipass(a, b, settings)

    assert np.array_equal(mask_s, mask_r)
    assert np.array_equal(mask_r, mask_a)
    assert np.allclose(u_s, u_r, equal_nan=True, atol=1e-8)
    assert np.allclose(v_s, v_r, equal_nan=True, atol=1e-8)


def test_invalid_backend_raises(image_pair):
    """Verify that specifying an unsupported backend raises ValueError."""
    im1, im2 = image_pair
    a = im1[:64, :64].astype(np.int32)
    b = im2[:64, :64].astype(np.int32)

    with pytest.raises(ValueError, match="Unknown backend"):
        pyprocess.extended_search_area_piv(a, b, window_size=32, overlap=16, backend="nonexistent_backend")


def test_rust_backend_missing_raises_importerror(image_pair, monkeypatch):
    """Verify that requesting backend='rust' when openpiv_rust is unavailable raises ImportError, while auto falls back."""
    im1, im2 = image_pair
    a = im1[:64, :64].astype(np.int32)
    b = im2[:64, :64].astype(np.int32)

    monkeypatch.setattr(pyprocess, "HAS_RUST", False)
    with pytest.raises(ImportError, match="openpiv_rust is not installed"):
        pyprocess.extended_search_area_piv(a, b, window_size=32, overlap=16, backend="rust")

    # In contrast, auto backend gracefully falls back to scipy without raising
    u, v, s2n = pyprocess.extended_search_area_piv(a, b, window_size=32, overlap=16, backend="auto")
    assert u.shape == (3, 3)
    assert not np.isnan(u).all()

