import sys
import time
import numpy as np
from openpiv import pyprocess
import openpiv_rust

print('=== OpenPIV Rust vs scipy.fft Benchmark & Validation ===', flush=True)

# Shapes representing real PIV interrogation window batches
shapes = [
    (660, 32, 32, 'window_size=32 (medium batch)'),
    (2145, 64, 64, 'window_size=64 (fine pass / large batch)'),
    (143, 128, 128, 'window_size=128 (coarse pass / large windows)'),
]

rng = np.random.default_rng(42)

for N, H, W, label in shapes:
    print(f'\n--- Testing {label}: shape=({N}, {H}, {W}) ---', flush=True)
    a = rng.random((N, H, W), dtype=np.float64)
    b = rng.random((N, H, W), dtype=np.float64)

    # 1. Correctness check (raw correlation without python normalization overhead)
    scipy_out = pyprocess.fft_correlate_images(a, b, correlation_method='circular', normalized_correlation=False)
    rust_out = openpiv_rust.fft_correlate_circular(a, b, normalized_correlation=False)

    max_diff = np.max(np.abs(scipy_out - rust_out))
    is_close = np.allclose(scipy_out, rust_out, atol=1e-5)
    print(f'Numerical validation: max_diff = {max_diff:.2e} | matches np.allclose: {is_close}', flush=True)

    # 2. Timing benchmark
    reps = 10

    # Warmup
    for _ in range(2):
        pyprocess.fft_correlate_images(a, b, correlation_method='circular', normalized_correlation=False)
        openpiv_rust.fft_correlate_circular(a, b, normalized_correlation=False)

    t0 = time.perf_counter()
    for _ in range(reps):
        scipy_res = pyprocess.fft_correlate_images(a, b, correlation_method='circular', normalized_correlation=False)
    t_scipy = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    for _ in range(reps):
        rust_res = openpiv_rust.fft_correlate_circular(a, b, normalized_correlation=False)
    t_rust = (time.perf_counter() - t0) / reps

    print(f'  scipy.fft (C++ PocketFFT): {t_scipy * 1000:7.2f} ms', flush=True)
    print(f'  Rust (Rayon + RustFFT):    {t_rust * 1000:7.2f} ms', flush=True)
    print(f'  Speedup vs scipy.fft:      {t_scipy / t_rust:7.2f}x', flush=True)

print('\nBenchmark completed.', flush=True)
