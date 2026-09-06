"""Benchmark FFT backends across many repeated calls of the SAME shape,
mimicking a batch job processing many image pairs with fixed window sizes
per pass -- the scenario where FFTW's MEASURE/wisdom planning should pay off
its upfront planning cost. Throwaway script for explore/fft-backends.
"""
import time
import numpy as np

N_CALLS = 40  # ~ processing 40 image pairs at one pass/window size


def main():
    rng = np.random.default_rng(0)
    N, h, w = 660, 31, 31  # windowsize=32 batch, a common case
    pairs = [(rng.random((N, h, w)), rng.random((N, h, w))) for _ in range(N_CALLS)]

    import scipy.fft as spfft

    def run_scipy(a, b):
        fa = spfft.rfft2(a, axes=(-2, -1))
        fb = spfft.rfft2(b, axes=(-2, -1))
        return spfft.irfft2(np.conj(fa) * fb)

    t0 = time.perf_counter()
    for a, b in pairs:
        run_scipy(a, b)
    t_scipy = time.perf_counter() - t0
    print(f"scipy.fft:                 {t_scipy*1000:8.1f} ms total, {t_scipy/N_CALLS*1000:6.2f} ms/call")

    import pyfftw

    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(30)

    def run_pyfftw_estimate(a, b):
        fa = pyfftw.interfaces.numpy_fft.rfft2(a, axes=(-2, -1), threads=1)
        fb = pyfftw.interfaces.numpy_fft.rfft2(b, axes=(-2, -1), threads=1)
        return pyfftw.interfaces.numpy_fft.irfft2(np.conj(fa) * fb, threads=1)

    t0 = time.perf_counter()
    for a, b in pairs:
        run_pyfftw_estimate(a, b)
    t_estimate = time.perf_counter() - t0
    print(f"pyfftw (ESTIMATE, cached): {t_estimate*1000:8.1f} ms total, {t_estimate/N_CALLS*1000:6.2f} ms/call")

    # Build explicit MEASURE-planned FFTW objects once, reuse the buffers for
    # every call -- this is the "amortize planning across a batch job" path.
    a_in = pyfftw.empty_aligned((N, h, w), dtype='float64')
    b_in = pyfftw.empty_aligned((N, h, w), dtype='float64')
    fshape = (N, h, w // 2 + 1)
    a_out = pyfftw.empty_aligned(fshape, dtype='complex128')
    b_out = pyfftw.empty_aligned(fshape, dtype='complex128')
    corr_in = pyfftw.empty_aligned(fshape, dtype='complex128')
    corr_out = pyfftw.empty_aligned((N, h, w), dtype='float64')

    t0 = time.perf_counter()
    fft_a = pyfftw.FFTW(a_in, a_out, axes=(-2, -1), direction='FFTW_FORWARD', flags=('FFTW_MEASURE',), threads=1)
    fft_b = pyfftw.FFTW(b_in, b_out, axes=(-2, -1), direction='FFTW_FORWARD', flags=('FFTW_MEASURE',), threads=1)
    ifft_c = pyfftw.FFTW(corr_in, corr_out, axes=(-2, -1), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE',), threads=1)
    plan_time = time.perf_counter() - t0
    print(f"pyfftw MEASURE plan build: {plan_time*1000:8.1f} ms (one-time)")

    def run_pyfftw_measure(a, b):
        a_in[:] = a
        b_in[:] = b
        fft_a()
        fft_b()
        corr_in[:] = np.conj(a_out) * b_out
        ifft_c()
        return corr_out.copy()

    t0 = time.perf_counter()
    for a, b in pairs:
        run_pyfftw_measure(a, b)
    t_measure = time.perf_counter() - t0
    print(f"pyfftw (MEASURE, reused):  {t_measure*1000:8.1f} ms total, {t_measure/N_CALLS*1000:6.2f} ms/call "
          f"(+ {plan_time*1000:.1f} ms one-time plan)")

    print()
    print(f"speedup vs scipy.fft: pyfftw ESTIMATE = {t_scipy/t_estimate:.2f}x, "
          f"pyfftw MEASURE (amortized over {N_CALLS} calls) = {t_scipy/(t_measure+plan_time):.2f}x, "
          f"pyfftw MEASURE (steady-state only) = {t_scipy/N_CALLS/(t_measure/N_CALLS):.2f}x")


if __name__ == "__main__":
    main()
