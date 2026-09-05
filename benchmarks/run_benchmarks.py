"""OpenPIV Performance & Backend Benchmarking Suite

Run this script to benchmark cross-correlation, subpixel peak finding,
and end-to-end multi-pass windef deformation across available backends.

Usage:
    python benchmarks/run_benchmarks.py
"""
import time
import numpy as np
from openpiv import pyprocess, windef
from openpiv.settings import PIVSettings

try:
    import openpiv_rust
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

def benchmark_correlation():
    print("\n" + "=" * 60)
    print(" 1. CROSS-CORRELATION BENCHMARK (225 windows of 32x32)")
    print("=" * 60)
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
        print("  Rust backend:   NOT INSTALLED (run 'maturin develop' in crates/openpiv_rust)")

def benchmark_subpixel():
    print("\n" + "=" * 60)
    print(" 2. SUBPIXEL PEAK POSITION BENCHMARK (961 windows)")
    print("=" * 60)
    rng = np.random.default_rng(42)
    corr = rng.random((961, 33, 33), dtype=np.float64)

    t0 = time.perf_counter()
    for _ in range(5):
        pyprocess.correlation_to_displacement(corr, 31, 31, subpixel_method="gaussian")
    t_py = (time.perf_counter() - t0) * 200

    print(f"  Active displacement engine: {t_py:.2f} ms for 961 windows")
    if HAS_RUST:
        print("  (Using parallel Rust subpixel peak finder via openpiv_rust)")
    else:
        print("  (Using Python loop fallback)")

def benchmark_windef():
    print("\n" + "=" * 60)
    print(" 3. END-TO-END MULTIGRID WINDEF (3 passes on 256x256 frames)")
    print("=" * 60)
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

if __name__ == "__main__":
    print(f"OpenPIV Benchmark Suite | Rust Backend Available: {HAS_RUST}")
    benchmark_correlation()
    benchmark_subpixel()
    benchmark_windef()
    print("\n" + "=" * 60)
    print(" Benchmarks Completed Successfully!")
    print("=" * 60 + "\n")
