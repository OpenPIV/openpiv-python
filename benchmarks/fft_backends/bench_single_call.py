"""Benchmark FFT backends for openpiv's batched cross-correlation pattern.

Not part of the package; a throwaway script for the explore/fft-backends
branch. Run with: uv run python scripts_bench_fft.py
"""
import time
import numpy as np


def bench(a, b, run, reps=5):
    run(a, b)  # warmup / JIT
    run(a, b)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = run(a, b)
    dt = (time.perf_counter() - t0) / reps
    return dt, out


def make_backend_numpy():
    import numpy.fft as npfft

    def run(a, b):
        fa = npfft.rfft2(a, axes=(-2, -1))
        fb = npfft.rfft2(b, axes=(-2, -1))
        return npfft.irfft2(np.conj(fa) * fb)

    return run


def make_backend_scipy():
    import scipy.fft as spfft

    def run(a, b):
        fa = spfft.rfft2(a, axes=(-2, -1))
        fb = spfft.rfft2(b, axes=(-2, -1))
        return spfft.irfft2(np.conj(fa) * fb)

    return run


def make_backend_pyfftw():
    import pyfftw

    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(30)

    def run(a, b):
        fa = pyfftw.interfaces.numpy_fft.rfft2(a, axes=(-2, -1), threads=1)
        fb = pyfftw.interfaces.numpy_fft.rfft2(b, axes=(-2, -1), threads=1)
        return pyfftw.interfaces.numpy_fft.irfft2(np.conj(fa) * fb, threads=1)

    return run


def make_backend_rocketfft():
    # rocket-fft only patches numpy.fft *inside numba-jitted functions*.
    import numba

    @numba.njit(cache=True)
    def _corr(a, b):
        fa = np.fft.rfft2(a)
        fb = np.fft.rfft2(b)
        return np.fft.irfft2(np.conj(fa) * fb)

    def run(a, b):
        return _corr(a, b)

    return run


SHAPES = [
    (2145, 63, 63, "finest pass (windowsize=6, many small windows)"),
    (660, 31, 31, "windowsize=32"),
    (143, 127, 127, "windowsize=64, linear-padded"),
]


def main():
    rng = np.random.default_rng(0)
    for N, h, w, label in SHAPES:
        print(f"\n=== {label}: shape=({N},{h},{w}) ===")
        a = rng.random((N, h, w))
        b = rng.random((N, h, w))

        results = {}
        for backend_name, factory in [
            ("numpy.fft", make_backend_numpy),
            ("scipy.fft", make_backend_scipy),
            ("pyfftw", make_backend_pyfftw),
            ("rocket-fft(numba)", make_backend_rocketfft),
        ]:
            try:
                run = factory()
                dt, out = bench(a, b, run)
                results[backend_name] = (dt, out)
                print(f"  {backend_name:20s} {dt*1000:8.2f} ms")
            except Exception as e:  # noqa: BLE001 - benchmark script, report and continue
                print(f"  {backend_name:20s} FAILED: {e}")

        if "scipy.fft" in results:
            ref = results["scipy.fft"][1]
            for name, (dt, out) in results.items():
                ok = np.allclose(out, ref, atol=1e-6)
                base = results["scipy.fft"][0]
                print(f"    {name:20s} speedup vs scipy.fft: {base/dt:5.2f}x   matches: {ok}")


if __name__ == "__main__":
    main()
