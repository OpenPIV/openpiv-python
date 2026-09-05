"""OpenPIV Performance & Backend Benchmarking Suite

Run this script to benchmark cross-correlation, subpixel peak finding,
and end-to-end multi-pass windef deformation across available backends.

Usage:
    python benchmarks/run_benchmarks.py
"""
import time
import numpy as np
from openpiv import pyprocess, validation, windef
from openpiv.settings import PIVSettings

try:
    import openpiv_rust
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

def benchmark_windowing():
    print("\n" + "=" * 65)
    print(" 1. SLIDING WINDOW EXTRACTION (1024x1024 image -> 3,969 windows of 32x32)")
    print("=" * 65)
    rng = np.random.default_rng(42)
    img = rng.random((1024, 1024), dtype=np.float64)

    # Python / NumPy meshgrid
    t0 = time.perf_counter()
    for _ in range(20):
        pyprocess.sliding_window_array(img, (32, 32), (16, 16), backend="python")
    t_py = (time.perf_counter() - t0) / 20 * 1000

    print(f"  Python / NumPy: {t_py:.2f} ms")

    if HAS_RUST:
        t0 = time.perf_counter()
        for _ in range(20):
            pyprocess.sliding_window_array(img, (32, 32), (16, 16), backend="rust")
        t_rs = (time.perf_counter() - t0) / 20 * 1000
        print(f"  Parallel Rust:  {t_rs:.2f} ms (speedup: {t_py/t_rs:.2f}x)")
    else:
        print("  Parallel Rust:  NOT INSTALLED")


def benchmark_correlation():
    print("\n" + "=" * 65)
    print(" 2. BATCH CROSS-CORRELATION (225 windows of 32x32)")
    print("=" * 65)
    rng = np.random.default_rng(42)
    a = rng.random((225, 32, 32), dtype=np.float64)
    b = rng.random((225, 32, 32), dtype=np.float64)

    # Circular - SciPy
    t0 = time.perf_counter()
    for _ in range(10):
        pyprocess.fft_correlate_images(a, b, correlation_method="circular", backend="scipy", normalized_correlation=False)
    t_scipy_circ = (time.perf_counter() - t0) * 100

    # Linear - SciPy (64x64 power-of-2)
    t0 = time.perf_counter()
    for _ in range(10):
        pyprocess.fft_correlate_images(a, b, correlation_method="linear", backend="scipy", normalized_correlation=False)
    t_scipy_lin = (time.perf_counter() - t0) * 100

    print(f"  SciPy Circular: {t_scipy_circ:.2f} ms")
    print(f"  SciPy Linear:   {t_scipy_lin:.2f} ms")

    if HAS_RUST:
        t0 = time.perf_counter()
        for _ in range(10):
            pyprocess.fft_correlate_images(a, b, correlation_method="circular", backend="rust", normalized_correlation=False)
        t_rust_circ = (time.perf_counter() - t0) * 100

        t0 = time.perf_counter()
        for _ in range(10):
            pyprocess.fft_correlate_images(a, b, correlation_method="linear", backend="rust", normalized_correlation=False)
        t_rust_lin = (time.perf_counter() - t0) * 100

        print(f"  Rust Circular:  {t_rust_circ:.2f} ms (speedup: {t_scipy_circ/t_rust_circ:.2f}x)")
        print(f"  Rust Linear:    {t_rust_lin:.2f} ms (speedup: {t_scipy_lin/t_rust_lin:.2f}x)")
    else:
        print("  Rust backend:   NOT INSTALLED")


def benchmark_subpixel():
    print("\n" + "=" * 65)
    print(" 3. SUBPIXEL PEAK INTERPOLATION (961 windows, 33x33 plane)")
    print("=" * 65)
    rng = np.random.default_rng(42)
    corr = rng.random((961, 33, 33), dtype=np.float64)

    t0 = time.perf_counter()
    for _ in range(5):
        pyprocess.correlation_to_displacement(corr, 31, 31, subpixel_method="gaussian", backend="python")
    t_py = (time.perf_counter() - t0) / 5 * 1000

    print(f"  Python loop:    {t_py:.2f} ms")

    if HAS_RUST:
        t0 = time.perf_counter()
        for _ in range(20):
            pyprocess.correlation_to_displacement(corr, 31, 31, subpixel_method="gaussian", backend="rust")
        t_rs = (time.perf_counter() - t0) / 20 * 1000
        print(f"  Parallel Rust:  {t_rs:.2f} ms (speedup: {t_py/t_rs:.2f}x)")
    else:
        print("  Parallel Rust:  NOT INSTALLED")


def benchmark_sig2noise():
    print("\n" + "=" * 65)
    print(" 4. SIGNAL-TO-NOISE RATIO (961 windows, 63x63 correlation plane)")
    print("=" * 65)
    rng = np.random.default_rng(42)
    corr = rng.random((961, 63, 63), dtype=np.float64)

    # Peak-to-Peak
    t0 = time.perf_counter()
    for _ in range(5):
        pyprocess.sig2noise_ratio(corr, sig2noise_method="peak2peak", width=2, backend="python")
    t_py_p2p = (time.perf_counter() - t0) / 5 * 1000

    # Peak-to-Mean
    t0 = time.perf_counter()
    for _ in range(5):
        pyprocess.sig2noise_ratio(corr, sig2noise_method="peak2mean", width=2, backend="python")
    t_py_p2m = (time.perf_counter() - t0) / 5 * 1000

    print(f"  Python peak2peak: {t_py_p2p:.2f} ms")
    print(f"  Python peak2mean: {t_py_p2m:.2f} ms")

    if HAS_RUST:
        t0 = time.perf_counter()
        for _ in range(20):
            pyprocess.sig2noise_ratio(corr, sig2noise_method="peak2peak", width=2, backend="rust")
        t_rs_p2p = (time.perf_counter() - t0) / 20 * 1000

        t0 = time.perf_counter()
        for _ in range(20):
            pyprocess.sig2noise_ratio(corr, sig2noise_method="peak2mean", width=2, backend="rust")
        t_rs_p2m = (time.perf_counter() - t0) / 20 * 1000

        print(f"  Rust peak2peak:   {t_rs_p2p:.2f} ms (speedup: {t_py_p2p/t_rs_p2p:.2f}x)")
        print(f"  Rust peak2mean:   {t_rs_p2m:.2f} ms (speedup: {t_py_p2m/t_rs_p2m:.2f}x)")
    else:
        print("  Rust backend:     NOT INSTALLED")


def benchmark_median_validation():
    print("\n" + "=" * 65)
    print(" 5. NORMALIZED MEDIAN OUTLIER DETECTION (64x64 vector field)")
    print("=" * 65)
    rng = np.random.default_rng(42)
    u = rng.standard_normal((64, 64))
    v = rng.standard_normal((64, 64))

    t0 = time.perf_counter()
    for _ in range(5):
        validation.local_norm_median_val(u, v, ε=0.1, threshold=2.0, size=1, backend="scipy")
    t_scipy = (time.perf_counter() - t0) / 5 * 1000

    print(f"  SciPy reference: {t_scipy:.2f} ms")

    if HAS_RUST:
        t0 = time.perf_counter()
        for _ in range(50):
            validation.local_norm_median_val(u, v, ε=0.1, threshold=2.0, size=1, backend="rust")
        t_rs = (time.perf_counter() - t0) / 50 * 1000
        print(f"  Parallel Rust:   {t_rs:.2f} ms (speedup: {t_scipy/t_rs:.2f}x)")
    else:
        print("  Parallel Rust:   NOT INSTALLED")


def benchmark_extended_piv():
    print("\n" + "=" * 65)
    print(" 6. END-TO-END EXTENDED SEARCH AREA PIV (512x512, 961 windows)")
    print("=" * 65)
    rng = np.random.default_rng(42)
    a = rng.integers(0, 255, (512, 512), dtype=np.int32)
    b = rng.integers(0, 255, (512, 512), dtype=np.int32)

    t0 = time.perf_counter()
    for _ in range(5):
        pyprocess.extended_search_area_piv(a, b, window_size=32, overlap=16, backend="scipy")
    t_scipy = (time.perf_counter() - t0) / 5 * 1000

    print(f"  Full SciPy pipeline: {t_scipy:.2f} ms")

    if HAS_RUST:
        t0 = time.perf_counter()
        for _ in range(10):
            pyprocess.extended_search_area_piv(a, b, window_size=32, overlap=16, backend="rust")
        t_rs = (time.perf_counter() - t0) / 10 * 1000
        print(f"  Full Rust pipeline:  {t_rs:.2f} ms (speedup: {t_scipy/t_rs:.2f}x)")
    else:
        print("  Full Rust pipeline:  NOT INSTALLED")


def benchmark_windef():
    print("\n" + "=" * 65)
    print(" 7. END-TO-END MULTIGRID WINDEF (3 passes on 256x256 frames)")
    print("=" * 65)
    from openpiv.test.test_process import create_pair
    frame_a, frame_b = create_pair(image_size=256)

    settings = PIVSettings()
    settings.windowsizes = (64, 32, 16)
    settings.overlap = (32, 16, 8)
    settings.num_iterations = 3
    settings.sig2noise_validate = False
    settings.show_all_plots = False
    settings.show_plot = False

    # SciPy run
    settings.backend = "scipy"
    windef.multigrid_windef(frame_a, frame_b, settings) # warmup
    t0 = time.perf_counter()
    for _ in range(3):
        windef.multigrid_windef(frame_a, frame_b, settings)
    t_scipy = (time.perf_counter() - t0) / 3 * 1000

    print(f"  SciPy 3-pass Windef: {t_scipy:.2f} ms")

    if HAS_RUST:
        settings.backend = "rust"
        windef.multigrid_windef(frame_a, frame_b, settings) # warmup
        t0 = time.perf_counter()
        for _ in range(3):
            windef.multigrid_windef(frame_a, frame_b, settings)
        t_rust = (time.perf_counter() - t0) / 3 * 1000
        print(f"  Rust 3-pass Windef:  {t_rust:.2f} ms (speedup: {t_scipy/t_rust:.2f}x)")
    else:
        print("  Rust backend:        NOT INSTALLED")


if __name__ == "__main__":
    print(f"OpenPIV Benchmark Suite | Rust Acceleration Available: {HAS_RUST}")
    benchmark_windowing()
    benchmark_correlation()
    benchmark_subpixel()
    benchmark_sig2noise()
    benchmark_median_validation()
    benchmark_extended_piv()
    benchmark_windef()
    print("\n" + "=" * 65)
    print(" All Benchmarks Completed Successfully!")
    print("=" * 65 + "\n")
